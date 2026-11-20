// The clock worker's shared state and its small helpers: the model, the frame, the ledgers, the clock, the mode (T-2374: moved out of clock.worker.ts). Other clock worker modules read these bindings and write them through the `put` functions.

import { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { anchor, toLocal } from "../map/geo";
import { CITIES, cityById, cityWithin, HOME_CITY, NEW_GAME_FRAME, zoneCountry } from "../game/cities";
import { frameKindName, makeFrame, type Frame, type FrameSpec } from "../game/frame";
import { PERIODS } from "../game/periods";
import { LOAN_AMOUNTS, loanAllowed, loanBalance, loanProfit, newLoan, type Loan } from "../game/loan";
import { type LineShape } from "../game/lineShape";
import { type Fares } from "../game/fareCurve";
import { runFactorOf } from "../game/income";
import { DAY_KINDS, dayKindOf } from "../game/days";
import { kindOfIn, rowOfDay, rowOfIn, type DayRow, type HolidayData } from "../game/holidayCalendar";
import { shiftAt } from "../game/localTime";
import { between, lagged, table, type KindOf, type PerHour } from "../game/moneyTiming";
import { type PackedOutlines } from "../map/coarseOutlines";
import { NEW_LINE_CARS, START_CLOCK, type GameMode, type DayLedger, type HourLedger, type NewHeadways, type GuideSave, type Achieved } from "./saveFile";
import type { FromClock } from "./protocol";
import { loadFailKey } from "../game/loadText";
import { letGoSlot, saves } from "./slots";
import { sendTrips } from "./clockDemand";
import { startFares } from "./clockFares";
import { firstBuiltAt, lineFirstStops, moneyLag } from "./clockMain";
import { gameNow, moneyNow } from "./clockMoney";
import { afterLoadPanic, isPanic, kept, openFailReason } from "./clockPanic";

/** Run 0 of each line is resampled this often (m) for the renderer's line texture. */
export const SAMPLE_STEP = 50;
export const SNAPSHOT_DEBOUNCE_MS = 500;

/** Autosave (T-029): this long after the last one if anything changed, wall ms; at once after a
 * construct or other paid edit, and when the page is hidden. */
export const AUTOSAVE_MS = 60_000;
/** Days of money kept in the ledger (every one, for the History chart's all time: T-2143; old ones folded by month,
 * `foldLedger`), and hours in the hourly one (T-447). */
export const LEDGER_DAYS = Infinity;
export const LEDGER_HOURS = 48;
/** T-3521, T-3542: each period on each kind of day, weekdays' first, then weekends' (`TrackApi.running_cost_period`). */
export const TEN = Array.from({ length: 2 * PERIODS.length }, (_, k) => k);

export let api: TrackApi | null = null;
/** the world's home zone, written into saves and snapshots; never the map's focused zone */
export const city = HOME_CITY;
/** the frame of the loaded game (T-125): new games are in the world frame, saves made before it in
 * the legacy one; read from the track model after every load */
export let frame: Frame = makeFrame(NEW_GAME_FRAME);
export function frameSpec(): FrameSpec { return { kind: frame.kind, lon: frame.lon, lat: frame.lat }; }
export function syncFrame() {
  const o = api!.frame_origin();
  frame = makeFrame({ kind: frameKindName(api!.frame_kind()), lon: o[0], lat: o[1] });
}

export let epoch = NaN;

/** T-3686: a game is being loaded (`loadGame` to its `restart`): the track model holds no network or a half-made one
 * (empty while masks come in, then the save's before its last mask), so no state or snapshot of it goes out; one
 * that did was taken by demand as the loaded game's network (a zone in view solved with no lines) */
export let midLoad = false;
/** the last edit's time by step, ms (T-229): sent with its answer for `?debug=1` */
export let timing: Record<string, number> = {};

// Money and the clock (T-028). The main thread's clock is the game's; it tells this worker each
// time it is set, paused or changes speed, and the worker settles fares and running costs one game
// hour at a time from that.
export let fares: Fares = startFares();
export let ledger: DayLedger[] = [];
export let hourLedger: HourLedger[] = [];
/** fare income per game hour in each period, US$M (from the main thread's demand results), per
 * kind of day (T-260, game/days.ts) */
export let incomePerHour: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
/** the same per line id (T-293), for each line's fares taken today */
export let lineIncome = new Map<number, number[][]>();
/** train rides per game hour as `incomePerHour` (T-565), and per line id */
export let ridesPerHour: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
export let lineRides = new Map<number, number[][]>();
/** T-736: the boarding account's in-period shape strengths per kind of day and period, and the lag
 * (s) money lands after riders set out (game/moneyTiming.ts) */
export const peakC: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));

export const fareHour = (per: number[][] | undefined, tl: number, kindOf?: KindOf) => lagged(table(per), peakC, tl, moneyLag, 0, kindOf);
export const rideHour = (per: number[][] | undefined, tl: number, kindOf?: KindOf) => between(table(per), peakC, tl, tl + 3600, kindOf);

/** T-872: the holiday calendar (its own chunk, fetched once), and each line's country: its first
 * stop's zone's (`cityWithin`; "" outside every zone). Its holidays and own weekend decide
 * its kinds of day (T-262). */
export let holidayData: HolidayData | null = null;
export let lineCountries = new Map<number, string>();

export function placeLineCountries() {
  lineCountries = new Map([...lineFirstStops].map(([id, p]) => {
    const z = p ? cityWithin(p[0], p[1]) : null;
    return [id, z ? cityById(z).country ?? "" : ""];
  }));
}
/** T-1265: the coarse outlines name each station's zone (game/cities.ts `cityWithin`); the page's copy, posted
 * once after init (T-2802), and the lines' countries placed again when they are in */
export let outlinesIn: (p: PackedOutlines | null) => void = () => {};
export const outlinesGot = new Promise<PackedOutlines | null>((got) => { outlinesIn = got; });
export let outlinesRead: Promise<void> = Promise.resolve();
export let holidaysSent = "", rowsSent = "";
export const kindOfLine = (id: number): KindOf => kindOfIn(holidayData, lineCountries.get(id));
export function loadHolidays() {
  void import("../game/holidays.json").then((m) => {
    holidayData = m.default as unknown as HolidayData;
    if (syncHolidays()) sendTrips();
  }, () => {});
}
/** Send the track model each line's days whose kind is not the day number's, with their kind
 * (holidays, its country's own weekend, weekend days worked), from yesterday to 8 days ahead in
 * its local time; true if they changed. Cheap when nothing did: called before every read of the
 * boarding account by day. */
export function syncHolidays(): boolean {
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
  const rows = syncRows();
  if (key === holidaysSent) return rows;
  holidaysSent = key;
  api.set_line_holidays(new Float64Array(out));
  return true;
}
/** T-3521: send the track model each line's kind of each day (`TrackApi.set_line_rows`) where its
 * calendar is not the week's (public holidays, its country's own weekend, weekend days worked), from
 * yesterday to 8 days ahead in its local time, every day of that window for such a line, so its
 * window starts yesterday; true if they changed. */
function syncRows(): boolean {
  if (!api || !holidayData) return false;
  const now = gameNow(), out: number[] = [];
  for (const [id, c] of lineCountries) {
    if (!c) continue;
    const rowOf = rowOfIn(holidayData, c), d0 = Math.floor((now + api.line_shift(id)) / 86400);
    const days: number[] = [];
    let own = false;
    for (let d = d0 - 1; d <= d0 + 8; d++) { const r = rowOf(d); own ||= r !== rowOfDay(d); days.push(d, r); }
    if (own) out.push(id, days.length / 2, ...days);
  }
  const key = out.join();
  if (key === rowsSent) return false;
  rowsSent = key;
  api.set_line_rows(new Float64Array(out));
  return true;
}
/** A line's kind of each local day, weekdays or weekends (game/holidayCalendar.ts `rowOfIn`). */
export const rowOfLine = (id: number) => rowOfIn(holidayData, lineCountries.get(id));
// T-835: trains run from half a run before their period to half a run into it; `tail`, the cost a
// day a period adds by running to its own end before no service (`runTail`), spread over it.
// T-3521: `run` per hour by period and kind of day (T-3542: weekdays' five periods, then weekends';
// `TrackApi.running_cost_period`), each day's kind by `rowOf` (the week's unless given).
export const NP = PERIODS.length;
export const costHour = (run: number[], tl: number, tail: number[][] | undefined, rowOf: (day: number) => DayRow = rowOfDay) => lagged(((r, p) =>
  run[r * NP + p] + (tail?.[r]?.[p] ?? 0) / (PERIODS[p].to - PERIODS[p].from)) as PerHour, null, tl, moneyLag, moneyLag / 2, rowOf);
/** T-835 (service.rs `period_points`): a period followed by no service runs to its own end, not
 * half a run before it, so its line runs that much longer a day at its level; `run` per hour by
 * period and kind of day as `costHour`'s, `tph` trains an hour likewise, from `lines_info` entry `i`.
 * Per kind of day and period, US$M a day (the period after the day's last taken on the same kind). */
export function runTail(li: ArrayLike<number>, i: number, run: number[], tph: number[]): number[][] {
  const lead = Math.min(7200, Math.max(0, (li[i + 11] - 2 * li[i + 20]) / 4));
  return [0, 1].map((r) => PERIODS.map((p, q) => {
    const next = PERIODS.findIndex((x) => x.from === p.to % 24);
    const nextTph = next >= 0 ? tph[r * NP + next] : 0;
    return nextTph > 0 ? 0 : (run[r * NP + q] * lead) / 3600;
  }));
}

/** T-531: each running line's fares and running cost by region, [region, share] (game/regionMoney.ts) */
export let lineShares = new Map<number, { fare: [string, number][]; ride: [string, number][]; run: [string, number][] }>();
/** T-867: how the network's upkeep divides between regions, [region, share] (game/regionMoney.ts) */
export let upkeepRegions: [string, number][] = [];
/** T-2669: each line's running cost at its countries' prices (`runFactorOf` on its `run` shares, as
 * game/regionMoney.ts `lineRunFactor`), given to the track model, which drains the cash by it. Again
 * after each state is built, since a load clears them there. */
let runFactorsSent = "";
/** Returns whether they changed. */
export function applyRunFactors(): boolean {
  if (!api) return false;
  const pairs: number[] = [];
  for (const [id, s] of lineShares) {
    const f = runFactorOf(s.run, zoneCountry);
    if (f !== 1) pairs.push(id, f);
  }
  api.set_line_run_factors(new Float64Array(pairs));
  const key = pairs.join();
  if (key === runFactorsSent) return false;
  runFactorsSent = key;
  return true;
}
/** T-684: each running line's points per game hour by region and kind of day and period (game/points.ts) */
export let linePoints = new Map<number, [string, number[][]][]>();
/** T-685: each region's star, the game second it was earned */
export let stars = new Map<string, number>();
/** T-1437: each region's higher goals, by share (%), the game second each was reached */
export let goals = new Map<string, Record<string, number>>();
/** T-2124: the achievement steps earned; null for an older save until the page has worked them out */
export let achievements: Map<string, Achieved> | null = new Map();
/** T-706: each line's station note and the city it was chosen in */
export let lineNotes = new Map<string, [number, string]>();
/** T-3319: the bullet shape of each line the player gave one, by line id (game/lineShapeState.ts) */
export let lineShapes = new Map<string, LineShape>();
/** T-3420: the lines the player hid on the map, by line id (game/lineVisibility.ts) */
export let hiddenLines = new Set<string>();
/** T-3409: the zones the schematic is of (none: the whole network), and the station nodes and bus stops the page says are
 * in them (diagram/cut.ts) */
export let diagramCities: string[] = [];
export let diagramInside = { stations: new Set<number>(), stops: new Set<number>() };
/** T-1094: the train length last set on any line (0 for the most the platforms fit), a new line's start */
export let newLineCars = NEW_LINE_CARS;
/** T-3543: the headways (seconds at each level) a new line starts with, set from the timetables window; null: the game's counts */
export let newLineHeadway: NewHeadways = {};
/** T-3588: the weekend percent "Apply to all lines" last set, which new lines start with; null: none set (100) */
export let newLineWeekendShare: number | null = null;
/** T-2926: a new tram line's sections, and the most a tram has (an 80 m platform) */
export const TRAM_NEW_CARS = 2, TRAM_MAX_CARS = 4;
/** T-1822: the Getting started card's progress in this game; undefined: none. `guideWanted`: the page
 * says a new normal game starts with it. */
export let guide: GuideSave | undefined;
export let guideWanted = false;
/** T-2395: where the game opens (`SaveHeader.home`); undefined: none */
export let home: string | undefined;
/** T-3115: the Community share this game is a copy of (`SaveHeader.copy`); undefined: the player's own */
export let copyOf: string | undefined;
/** T-3707: why this page's games are modded (the "modded" message), and the game in hand's reasons (`SaveHeader.modded`):
 * its own and the page's, never fewer; undefined for the published game's own */
export let pageModded: string[] = [];
export let modded: string[] | undefined;
export function withModded(own: unknown): string[] | undefined {
  const all = [...new Set([...(Array.isArray(own) ? own.filter((x): x is string => typeof x === "string") : []), ...pageModded])].slice(0, 16);
  return all.length ? all : undefined;
}
/** T-531: paid edits' charges, held under "" until the main thread says how each divides */
export const heldCharges = new Map<number, { day: number; hour: number; build: number; trains: number }>();
/** Wall ms on the clock both threads read (`ToClock` "clock"): Date.now() drifts from it, and the
 * worker running ahead of the main thread's clock pulled money time back across midnight (T-488). */
export const wallNow = () => performance.timeOrigin + performance.now();
export let clk = { t: START_CLOCK, rate: 0, at: wallNow() };

/** T-284: money is settled up to this game second, not by whole hours, so the cash moves as fares
 * land. `partHour` is the hour being settled and `partFare`, `partCost` the shares of its fares and
 * running cost already in the cash (each reaches 1 as the hour ends, so an hour still pays its whole
 * fare and cost). */
export let settledT = START_CLOCK;
export let partHour = Math.floor(START_CLOCK / 3600);
export let partFare = (START_CLOCK % 3600) / 3600, partCost = partFare;
/** per game hour (absolute index), the share of its fares paid by each second into it (3601
 * values from 0 to 1), from the boarding account's trains leaving platforms with riders; an hour
 * without one pays evenly (T-284) */
export const hourShape = new Map<number, Float32Array>();

/** T-3788: a load into the running game failed (the slot is untouched): the same ending as an open that failed */
export function failInPlace(err: unknown, name: string) {
  const panicked = isPanic(err) ? afterLoadPanic(err) : null;
  const slot = saves.activeSlot;
  letGoSlot();
  send({ kind: "openFailed", from: "slot", slot, name, ...openFailReason(err, panicked, loadFailKey(String(err))) });
  if (panicked === null) console.error("account copy did not load:", err);
}
/** T-728: the game's mode, start zone and opened zones, kept in the save's header (`opened` absent
 * in a save from before it) */
export let mode: GameMode = "normal";
export let startZone: string | undefined;
/** T-2271: seconds the start zone's local time runs ahead of the clock (0 with no start zone: the
 * clock's own, New York's). The ledgers' days run midnight to midnight there, so the Money tab's day
 * turns as its clock passes midnight; their hours stay the clock's. */
export let dayShift = 0;
/** T-3407: a game with no start zone takes its first built station's, once; saved as `SaveHeader.dayShift`
 * so days booked under it keep their keys on a reload (the first station can be deleted later). */
export let dayPinned = false;
export const ledgerDay = (t: number) => Math.floor((t + dayShift) / 86400);
export function setStartZone(z: string | undefined, saved?: number) {
  startZone = z;
  const c = z ? CITIES.find((x) => x.id === z) : undefined;
  dayPinned = !c && saved !== undefined;
  dayShift = c ? shiftAt(c.center[0], c.center[1]) : dayPinned ? saved! : 0;
}
export function pinDayShift() {
  if (startZone || dayPinned || !firstBuiltAt) return;
  dayPinned = true;
  dayShift = shiftAt(firstBuiltAt[0], firstBuiltAt[1]);
}
export let opened: string[] | undefined = [];
/** T-1223: the zones demand holds fine, from the page; T-2153 the last two builds' zones */
export let demandFine: string[] | undefined;
export let demandBuilt: string[][] | undefined;
export function setMode(m: GameMode) {
  mode = m;
  api!.set_unlimited(m === "sandbox");
  if (m === "sandbox") loan = null;
}
/** T-1661, T-2022: the loan out, null for none */
export let loan: Loan | null = null;
/** A loan, or paying off what is left of it (r 0 done, else refused: a sandbox, a loan already out, a size that is not offered, payments past the profits (T-1906), no loan, too little cash). */
export function loanOp(op: { op: "borrow"; amount: number } | { op: "repay" }): number {
  const a = api!;
  if (mode === "sandbox") return 1;
  if (op.op === "borrow") {
    if (loan || !LOAN_AMOUNTS.includes(op.amount)) return 1;
    if (!loanAllowed(op.amount, loanProfit(ledger, ledgerDay(moneyNow())))) return 1;
    loan = newLoan(op.amount);
    a.set_cash(a.cash() + op.amount);
  } else {
    if (!loan) return 1;
    const left = loanBalance(loan);
    if (a.cash() < left) return 1;
    a.set_cash(a.cash() - left);
    loan = null;
  }
  if (kept) kept.cash = a.cash();
  return 0;
}
/** T-3478: the first load is opening a file or backup picked before the reload, which is in no slot until it opens */
export let initIncoming = false;
/** the empty network's save, for a new game (keeps the water mask) */
export let emptyTrack: Uint8Array | null = null;
/** the sim's money parameters (`money_params()`: an array by index) by name, the names of the
 * `MoneyState` fields they fill; the first four are known before the model is up */
export function pricesOf(p: ArrayLike<number>) {
  return {
    carPrice: p[0], runCostCarKm: p[1], economy: p[2], runCostTrainKm: p[3], hsCarPrice: p[4], hsRunCostCarKm: p[5], hsRunCostTrainKm: p[6],
    upkeepTrackKm: p[7], upkeepStation: p[8], upkeepSingle: p[9], upkeepHsr: p[10], busPrice: p[11], runCostBusKm: p[12], upkeepBusKm: p[13],
    busRouteKm: p[14], upkeepViaductKm: p[15], upkeepCutKm: p[16], upkeepBoredKm: p[17], boatPrice: p[18], runCostBoatKm: p[19], upkeepPort: p[20], portPrice: p[21],
    upkeepStationUnder: p[22], upkeepOnePlatform: p[23],
  };
}
export let prices = pricesOf([2.5, 2, 100, 9]);

export let freezePort: MessagePort | undefined;
export let freezeMemory: WebAssembly.Memory | undefined;
export let freezeTimer: ReturnType<typeof setInterval> | undefined;
export function freezeCheckpoint(event: string, detail: unknown = null) {
  if (!freezePort) return;
  try { freezePort.postMessage({ event, detail, wasmMB: freezeMemory ? Math.round(freezeMemory.buffer.byteLength / 1048576) : undefined }); }
  catch { /* diagnostics must not affect the game */ }
}
export function send(msg: FromClock, transfer: Transferable[] = []) {
  if (msg.kind === "edited") freezeCheckpoint("edited", { seq: msg.seq, ok: msg.ok, ms: msg.ms });
  if (msg.kind === "error") freezeCheckpoint("error", msg.message);
  postMessage(msg, { transfer });
}

/** world metres -> lng/lat in the loaded game's frame (game/frame.ts) */
export function packLngLat(x: number, y: number): [number, number] {
  return frame.toLonLat(x, y);
}
/** world metres -> local units, relative to the draw anchor the main thread last sent (T-307) */
export function packLocal(x: number, y: number): [number, number] {
  const [lng, lat] = packLngLat(x, y);
  return toLocal(lng, lat);
}
export function anchorSpec() {
  const a = anchor();
  return { lng: a.lng, lat: a.lat, version: a.version };
}

export let messages = Promise.resolve();

/** T-1039: the network's version (any edit or load moves it on), and the route request whose
 * preview found every mask its geometry needs in, at that version: its edit needs no second look */
export let trackVersion = 0;
export function putAchievements(v: typeof achievements) { achievements = v; }
export function putApi(v: typeof api) { api = v; }
export function putClk(v: typeof clk) { clk = v; }
export function putCopyOf(v: typeof copyOf) { copyOf = v; }
export function putDemandBuilt(v: typeof demandBuilt) { demandBuilt = v; }
export function putDemandFine(v: typeof demandFine) { demandFine = v; }
export function putDiagramCities(v: typeof diagramCities) { diagramCities = v; }
export function putDiagramInside(v: typeof diagramInside) { diagramInside = v; }
export function putEmptyTrack(v: typeof emptyTrack) { emptyTrack = v; }
export function putEpoch(v: typeof epoch) { epoch = v; }
export function putFares(v: typeof fares) { fares = v; }
export function putFrame(v: typeof frame) { frame = v; }
export function putFreezeMemory(v: typeof freezeMemory) { freezeMemory = v; }
export function putFreezePort(v: typeof freezePort) { freezePort = v; }
export function putFreezeTimer(v: typeof freezeTimer) { freezeTimer = v; }
export function putGoals(v: typeof goals) { goals = v; }
export function putGuide(v: typeof guide) { guide = v; }
export function putGuideWanted(v: typeof guideWanted) { guideWanted = v; }
export function putHiddenLines(v: typeof hiddenLines) { hiddenLines = v; }
export function putHolidaysSent(v: typeof holidaysSent) { holidaysSent = v; }
export function putHome(v: typeof home) { home = v; }
export function putHourLedger(v: typeof hourLedger) { hourLedger = v; }
export function putIncomePerHour(v: typeof incomePerHour) { incomePerHour = v; }
export function putInitIncoming(v: typeof initIncoming) { initIncoming = v; }
export function putLedger(v: typeof ledger) { ledger = v; }
export function putLineIncome(v: typeof lineIncome) { lineIncome = v; }
export function putLineNotes(v: typeof lineNotes) { lineNotes = v; }
export function putLinePoints(v: typeof linePoints) { linePoints = v; }
export function putLineRides(v: typeof lineRides) { lineRides = v; }
export function putLineShapes(v: typeof lineShapes) { lineShapes = v; }
export function putLineShares(v: typeof lineShares) { lineShares = v; }
export function putLoan(v: typeof loan) { loan = v; }
export function putMessages(v: typeof messages) { messages = v; }
export function putMidLoad(v: typeof midLoad) { midLoad = v; }
export function putModded(v: typeof modded) { modded = v; }
export function putNewLineCars(v: typeof newLineCars) { newLineCars = v; }
export function putNewLineHeadway(v: typeof newLineHeadway) { newLineHeadway = v; }
export function putNewLineWeekendShare(v: typeof newLineWeekendShare) { newLineWeekendShare = v; }
export function putOpened(v: typeof opened) { opened = v; }
export function putOutlinesRead(v: typeof outlinesRead) { outlinesRead = v; }
export function putPageModded(v: typeof pageModded) { pageModded = v; }
export function putPartCost(v: typeof partCost) { partCost = v; }
export function putPartFare(v: typeof partFare) { partFare = v; }
export function putPartHour(v: typeof partHour) { partHour = v; }
export function putPrices(v: typeof prices) { prices = v; }
export function putRidesPerHour(v: typeof ridesPerHour) { ridesPerHour = v; }
export function putRowsSent(v: typeof rowsSent) { rowsSent = v; }
export function putSettledT(v: typeof settledT) { settledT = v; }
export function putStars(v: typeof stars) { stars = v; }
export function putTiming(v: typeof timing) { timing = v; }
export function putTrackVersion(v: typeof trackVersion) { trackVersion = v; }
export function putUpkeepRegions(v: typeof upkeepRegions) { upkeepRegions = v; }
