// Main-thread side of the shared-world demand pool (T-116/T-117).
//
// - Loads only the packs a game needs (T-149): the regional market, the viewed city first, and
//   every city a built station can reach (workers/cityReach.ts); a city viewed later loads then.
//   A solve waits until every city it needs is open in every worker, so no constructed city's
//   riders are ever left out of a published result. Each worker owns its copy of every
//   population; the pool budget applies to the whole world, not each city.
// - T-325: a zone no station reaches, viewed or shown beside the view, opens `bare` in the aux
//   worker alone. Its city figures and bubbles come from there (no rail: walking and driving
//   only), no solve covers it and none is started for it. A station built within reach opens it
//   in every worker, and the solves cover it from then on. T-351: at startup the viewed zone's
//   pack is fetched before the network's reach is known; it opens bare if the reach is known by
//   the time it has downloaded and leaves the zone out, else in every worker as before. A new
//   game (`?new=1`) has no stations, so its reach is known at once and the viewed zone opens bare.
// - T-301: packs (and the market) nobody needs any more close in every worker (`sweep`).
// - Reports what it is loading (game/loading.ts `loadProgress`) and releases a loaded game's
//   clock hold once that game's network has its final result (HOLD_ROUND, T-347), or shows its
//   cached result (T-2560).
// - The day's five periods are split over the pool by index (worker i solves periods i, i + n,
//   ...). Each worker sends the free-flow estimate of all its periods first, then the crowding
//   rounds.
// - `submitNetwork` takes the clock worker's NetworkSnapshots (already debounced there, SPEC 6.5).
//   With `?demandTest=1` the pool solves T-005's hand-made network instead (workers/demandTest.ts)
//   and ignores the clock's.
// - Never blocks: a newer network supersedes a running solve at the workers' next period
//   boundary; game/demand.ts keeps the last finished result until every period of the new one
//   has landed, then swaps it in whole (free flow first, again after each crowding round).

import { t } from "../i18n";
import { ownModesFrom } from "../game/corridorFigures";
import { batch, effect } from "@preact/signals";
import { activeCity, CITIES, cityById, getWorldFrame, HOME_CITY, isCity, LEGACY_FRAME, MARKET_FRAME, viewedZone, worldFrameEpoch, worldToCity, cityToMarket, type CityId } from "../game/cities";
import type { FrameSpec } from "../game/frame";
import { splitOf } from "../game/types";
import { BACK_DAY, DEMAND_DAYS, OUT_DAY, PLAIN_KIND, rushRun, type DayKind, type DemandDay } from "../game/days";
import { display, marketOff, selection, shownDay, tool } from "../game/ui";
import {
  cityDemandViews, demandByDay, intercityDemand, mainDemandByDay,
  type DayDemand,
  type DemandView,
  demandStep,
  demandWork,
  editHeld,
  solveRestarted,
  type WorkPart,
  demandScope,
  demandUpdating,
  networkWorld,
  finalWorld,
  demandBehindFrom,
  demandView,
  setDemandQueries,
  type CellModes,
  type CellRiders,
  type CityCells,
  type ComplexTransfers,
  type TransferPair,
  type Flows,
  type IntercitySelection,
  type IntercityStation,
  INTERCITY_PLACES,
  type StationRiders,
  type ZoneRiders,
  type Corridors,
  type CorridorBook,
  type CorridorRows,
  NO_CORRIDORS,
  heldRows,
  rowsCanCome,
  rowsNow,
  seedRows,
  setRowsSource,
  demandSolving,
  roughZones,
  refiningZone,
  focusZones,
  demandAskTurn,
  editsSettling,
  lineLoadsFor,
  demandHeld,
} from "../game/demand";
import { FOCUS_WORKERS, FocusPool } from "./demandFocus";
import { ASIDE_MAX_ZONES, SNAP_MAX_BYTES, ShownStore, type AsideAnswer, type AsideShown, type Snapshot } from "./demandShown";
import { tracing } from "../game/freezeTrace";
import { demandNote, startDemandTrace } from "./demandTrace";
import { addressLimited, deviceGB, heapStep, memBudget, mobileDevice, phoneLoad, smallDevice } from "./deviceMemory";
import { outsideWasmBytes } from "./liveMemory";
import { isMemoryFailure } from "../game/loadText";
import { makeFrame } from "../game/frame";
import { editLeftOf, editPrior, noteEditSolve } from "../game/editLeft";
import { choosing, demandIdle, endCardFull, noteLoad, noteLoadWorkers, gameLoadArmed, gameLoadSeq, viewFiguresIn, gameWait, loadComplete, loadFailed, loadProgress, lastStep, markAlive, markStep, memoryResolve, onLoadStop, onLoadStuck, setDownloadProbe, setUnitQuietProbe, solveLeft, startup, takeGameLoadHold, zoneFailures, zoneLoads, zoneSteps, type ZoneStep } from "../game/loading";
import { barPlace, type BarPlace, CACHE_SHOW_MS, GAME_MS, GUESS_SHARES, LATER_GUESS, lastWait, loadPace, nextLeft, rememberPace, rememberWait, sameSize, type Pace, sharesOf, solveStage, STAGE_TEXT, stagedFraction, stageInHand, STEP_CAP, timeShare, type StageMs, zoneShare } from "../game/loadBar";
import { linesSettled } from "../game/lineFigures";
import { clock } from "../game/clock";
import { PERIODS, TIMES_PER_STOP, periodsMatch } from "../game/periods";
import { EVERYDAY_KEYS, membersOf, type PurposeFilter } from "../game/purposes";
import { perfOff } from "../perfFlags";
import { cityReach, LEVEL, marketZonesNear } from "./demandClusters";
import { citiesReached, footprintOf, loadFootprints, REACH_M, zonesNearStations, type Footprint } from "./cityReach";
import { nearbyZones, zoneAt, zonesLoadForView } from "../game/zones";
import { packToLngLat } from "../game/coords";
import { addOpened, reportDemandFine } from "../game/openedZones";
import { memoryNeed } from "../game/memoryCheck";
import { builtKeep, firstFine, LEFT_ALONE_MS, LOD_COARSE, LOD_FINE, lodBudgetGB, lodMode, planFine, VIEW_SETTLE_MS, EDIT_KEEP_MS, zoneGB, zonesGB } from "./demandLod";
import { stopsById, world } from "../game/world";
import { stopFareScales } from "../game/money";
import { trackZones } from "../game/trackZones";
import { viewCentre } from "../game/listScope";
import { ALL_PERIODS, MARKET_POP, MARKET_QUERIES, ZONE_QUERIES, isQuery, type Answer, type SplitPlan, type SelSquares, type BareFigures, type DemandNetwork, type FromDemand, type Heap, type OpenRejected, type QueryKind, type ToDemand } from "./demandProtocol";
import { addParts, dayFactor, dayResultOf, dayViews, flowsBody, selectionFlows, signature, stationFlows, summaryOnDay, sumParts, unpackDays, type CorridorRail, type DayInput, type DayViews, type FlowsAnswer, type FromSums, type PackedDays, type PurposePart, type SumsAsk, type SumsCtx, type ToSums } from "./demandSums";
import { SPLIT, SUM_FLAGS } from "./exactSum";
import { Holding, MARKET_UNIT } from "./demandPlan";
import type { Box } from "./demandClusters";
import { primeDealt } from "../game/farChunks";
import { tripLengthsOf, type TripLengths } from "../game/tripLengths";
import { readMarketPeriod, readTripPeriod, setMarketTripQuery, setTripQuery, tripKey, type MarketPeriod, type RouteNodes, type Trip } from "../game/trip";
import { demandTestSnapshot } from "./demandTest";
import type { NetworkSnapshot } from "./protocol";
import { dataUrl } from "../game/dataBase";
import { fetchMarketDrives } from "../game/marketRule";
import { lineFareMul, ownLineFare, type FareArea, type Fares } from "../game/fareCurve";
import { takesMarketless } from "../game/fareInEffect";
import { localStep } from "./demandSteps";
import { saveSlotsState } from "./clockClient";
import { readDemandAhead, readDemandCache, writeDemandCache } from "./saveFile";
import { DEMAND_KEY, CACHE_FORMAT, CACHE_VERSION, cacheAvailable, canonical, fromStored, sameKey, sha256, standsIn, storedSize, toStored, type CacheKey, type StoredDemand } from "./demandCache";
import { patientFetch } from "./patientFetch";
import { binGzOf, isGzip, unpackBin } from "./packZip";

/**
 * Crowding rounds after the free-flow estimate (T-022, notes/T-022.md). Each lands as its own
 * update. One round leaves the worst segment at its most overloaded (the first round moves half
 * the riders a penalty pushes, all at once); by the third a player-sized network moves under 1% a
 * round and New York's real network under 10%.
 */
const ROUNDS = 3;
const PROGRESS_GAP_MS = 100;
/**
 * A load's solve is done, for the loading bar, once this round is published: the free-flow
 * estimate (T-172; a city opened by a pan or a station reaching it). The crowding rounds then
 * refine it in the background, as after any edit.
 */
const READY_ROUND = 0;
/**
 * A loaded game's clock runs, and its loading card goes, once the zone in view has its final: the
 * solve's last round, or (T-421) that of the zone's cluster solved ahead of the rest (T-402), the
 * other zones then showing their own loading bars until the solve's final (`released`). The cached
 * result of the last time the game was open (T-328) releases it once the network is in (T-2560:
 * `cacheDone`). A viewed zone open bare is final from its opening (T-351: `holdDone`).
 */
const HOLD_ROUND = ROUNDS;
const PERIOD_STRIDE = 6;

const FLOWS_FIRST_MS = 1000;
/** T-2746: the same wait with no solve running, a guard only: the far end's `flows` is posted within ~100 ms */
const FLOWS_IDLE_MS = 5000;
/** T-2136: a corridor rows question no solving worker has answered by then counts as none holding them
 * (a worker replaced meanwhile never answers); asked again with the next result */
const ROWS_WAIT_MS = 20000;
/** T-2136: one-place corridor rows kept with a cached result, the zone in view's first (about 80 kB each) */
const CACHED_ROWS = 16;

/** The first registry zone of each market place (a scan per place was 45 ms a query in China, T-1165). */
let placeCities: Map<number, (typeof CITIES)[number]> | null = null;
function cityOfPlace(place: number) {
  if (!placeCities) {
    placeCities = new Map();
    for (const c of CITIES) if (!placeCities.has(c.place)) placeCities.set(c.place, c);
  }
  return placeCities.get(place);
}

/**
 * The periods solving worker `i` of `n` owns: by index (`q % n`), unless that gives one worker both
 * periods of a demand level (2 workers: high demand's three), when they are dealt out level by level. An edit
 * to one level's trains re-solves only that level's periods (T-124), so they go to different workers.
 */
function periodsOfWorker(i: number, n: number): number[] {
  // T-3542: two workers deal the morning with the evening and night and the midday with the
  // afternoon: a rush edit (four rows: morning and afternoon) then waits on one period, high
  // demand's three still go two to one worker, and each worker has the same work as before
  // (`PERIOD_COST`); any more workers already part morning and afternoon by index
  if (n === 2 && PERIODS.length === 5) return i === 0 ? [0, 3, 4] : [1, 2];
  const qs = PERIODS.map((_, q) => q);
  const clash = qs.some((q) => qs.some((s) => s > q && PERIODS[s].demand === PERIODS[q].demand && s % n === q % n));
  const order = clash ? qs.slice().sort((a, b) => LEVEL[PERIODS[a].demand] - LEVEL[PERIODS[b].demand] || a - b) : qs;
  return order.filter((_, k) => k % n === i).sort((a, b) => a - b);
}

/**
 * T-1607: the periods whose market each of `n` solving workers solves in a solve with the barrier
 * (a load, a zone opened), the cities staying with `periodsOfWorker`. Periods differ in work and
 * five do not deal evenly, so a market may go to a worker with less to do; the two sides trade
 * loads every round. Chosen to make the busiest worker's round shortest, by each period's time a
 * round on all of Japan with its market 4/5 of it (notes/T-1607.md), then the fewest moved, and
 * only where that round is at least 11% shorter (in effect 3 workers: 2 measured no gain).
 * `?perfOff=share` keeps every market with its cities.
 */
const PERIOD_COST = [126, 125, 120, 120, 67], MARKET_SHARE = 0.8, SHARE_GAIN = 0.11;
export function marketsOfWorkers(n: number): number[][] {
  const cities = Array.from({ length: n }, (_, i) => periodsOfWorker(i, n));
  const ownerOf = PERIODS.map((_, q) => cities.findIndex((qs) => qs.includes(q)));
  let best = ownerOf.slice(), bestCost = Infinity, bestMoved = 0;
  if (n > 1 && n < PERIODS.length && !perfOff("share")) {
    const at = PERIODS.map(() => 0);
    for (let k = 0; k < n ** PERIODS.length; k++) {
      for (let q = 0, r = k; q < PERIODS.length; q++, r = Math.floor(r / n)) at[q] = r % n;
      const load = new Array(n).fill(0);
      let moved = 0;
      PERIODS.forEach((_, q) => {
        load[ownerOf[q]] += PERIOD_COST[q] * (1 - MARKET_SHARE);
        load[at[q]] += PERIOD_COST[q] * MARKET_SHARE;
        if (at[q] !== ownerOf[q]) moved++;
      });
      const cost = Math.max(...load);
      if (cost < bestCost - 1e-9 || (cost < bestCost + 1e-9 && moved < bestMoved)) { best = at.slice(); bestCost = cost; bestMoved = moved; }
    }
    const whole = Math.max(...cities.map((qs) => qs.reduce((s, q) => s + PERIOD_COST[q], 0)));
    if (bestCost > whole * (1 - SHARE_GAIN)) best = ownerOf.slice();
  }
  return cities.map((_, i) => PERIODS.map((_, q) => q).filter((q) => best[q] === i));
}

/**
 * Workers in the pool (T-361, T-381). By threads: 12 or more get 3, 6 to 11 get 2, fewer 1; then
 * by memory (each worker's WASM is a few hundred MB, notes/T-361.md): a phone or tablet, or 4 GB or less
 * (`smallDevice`), gets 1, and where memory is not reported (desktop Firefox, Safari) at most 2. The "Use more
 * cores" setting restores the older rule, half the threads minus one, 1 to 5 (one per period).
 * `?demandWorkers=N` (1-5) overrides everything. The pool is built once, so the setting counts at
 * the next load. T-2867: a 32-bit browser gets 1 whatever the setting (in 32-bit Firefox New York
 * reaches its final with 1 and runs out of memory with 2, notes/T-2867.md). T-3072: a pool that lost workers
 * to running out of memory stays that size for the tab's session (`rememberedPoolCap`).
 */
export function poolSize(): number {
  const q = workersAsked();
  if (q !== null) return q;
  if (addressLimited()) return 1;
  return Math.min(poolBySetup(), rememberedPoolCap());
}
/** T-3373: the solving workers a game whose workers each need `perWorker` bytes (WASM and JS) runs here: the
 * pool, less any past `SOLVING_SHARE` of the budget (`capPool`'s rule) */
export function workersFor(perWorker: number): number {
  const n = poolSize();
  if (!shareCapped()) return n;
  return poolIn(perWorker, n, solvingShare() * memBudget());
}
/** the most solving workers, 1 to `max`, whose WASM and JS together (`poolBytes`) fit `room` */
function poolIn(perWorker: number, max: number, room: number): number {
  let n = 1;
  while (n < max && poolBytes(perWorker, n + 1) <= room) n++;
  return n;
}
/** what `n` solving workers take together when one alone takes `perWorker` (WASM and JS): each holds everything, or
 * under `?split=1` (T-3373 B) each zone and the market have one holder, so a further worker adds only SPLIT_EXTRA */
export function poolBytes(perWorker: number, n: number): number {
  return SPLIT === "plan" ? perWorker + (n - 1) * SPLIT_EXTRA : n * perWorker;
}
/** `SOLVING_SHARE`, or SPLIT_SHARE under `?split=1` */
const solvingShare = () => (SPLIT === "plan" ? SPLIT_SHARE : SOLVING_SHARE);

const shareCapped = () => workersAsked() === null && !display.peek().moreCores;

/** `?demandWorkers=N` (1-5), else null */
function workersAsked(): number | null {
  const q = Number(new URLSearchParams(location.search).get("demandWorkers"));
  return Number.isInteger(q) && q >= 1 && q <= PERIODS.length ? q : null;
}
function poolBySetup(): number {
  const hc = navigator.hardwareConcurrency || 4;
  if (display.peek().moreCores) return Math.max(1, Math.min(PERIODS.length, Math.floor(hc / 2) - 1));
  if (smallDevice()) return 1;
  const byThreads = hc >= 12 ? 3 : hc >= 6 ? 2 : 1;
  return deviceGB() === undefined ? Math.min(byThreads, 2) : byThreads;
}

/** T-3072: the solving workers a worker's running out of memory left this tab with. Kept for the tab's
 * session (it survives the reload that "Play without intercity trips" does) and not for the browser:
 * what memory is free depends on what else is open, and a stale low count would slow the game for good. */
const POOL_CAP_KEY = "anitabuilder-pool-cap";
function rememberedPoolCap(): number {
  try { const n = Number(sessionStorage.getItem(POOL_CAP_KEY)); return Number.isInteger(n) && n >= 1 ? n : Infinity; } catch { return Infinity; }
}
/** (T-3933: also a load that took the tab down with it, game/oomRetry.ts) */
export function rememberPoolCap(n: number) {
  try { sessionStorage.setItem(POOL_CAP_KEY, String(Math.min(n, rememberedPoolCap()))); } catch { /* storage blocked */ }
}

/** The service demand reads, with the UI's ids. */
interface Service {
  frame: FrameSpec;
  fare: { base: number; perKm: number; nodes?: [number, number][]; scales?: Record<string, number>; byCity?: boolean; cityScales?: Record<string, number>; hsr?: number; ferry?: number; ferrySmall?: number; ferryLarge?: number; bus?: number; brt?: number; transfer?: number; lines?: Record<string, number>; zones?: FareArea[] };
  /** `complex`: the station's complex (T-161); absent = a complex of its own; `fare`: its fare zone's code (T-3740) */
  /** `level`: T-3888, the station's level where not at grade (0); `street`: T-4041, a tram stop on the street */
  stations: { id: string; x: number; y: number; level?: number; complex?: number; upgrades?: number; fare?: number; street?: boolean }[];
  /** `wrapS`: a circle's stand at its first stop, riding on round (T-798); `back`: run 1's stations
   * where a stop's way back is at a platform beside it (T-3348, `SnapshotLine.backStops`) */
  lines: { id: string; stops: string[]; back?: string[]; tph: number[]; hopS: number[][][]; hopKm?: number[][]; fast?: boolean; ferry?: boolean; boat?: 0 | 2; brt?: boolean; tram?: boolean; fareDist?: boolean; passFares?: number[][][]; dwellS: number[][][]; wrapS?: number; cars: number; span?: number[]; catches?: number[][] }[];
  /** station pairs joined by a passage (T-1281) */
  passages?: [string, string][];
  /** T-4048: station pairs the player separated */
  apart?: [string, string][];
  /** T-2410: stations no line stops at, left out of `stations`: their positions only reach zones
   * (opened ahead of a line on them); never in the key */
  alone?: [number, number][];
}

/** A network's stations as lon, lat pairs, the zones whose boxes they come near (T-404), and the
 * zones whose market riders reach any (T-396, T-377); `key`: the frame and positions they follow from */
interface Near { key: string; ll: Float64Array; near: CityId[]; markets: CityId[] }

/** A service's key: what demand reads of it (not `alone`). */
const keyOf = (svc: Service) => JSON.stringify({ ...svc, alone: undefined });
/** The stops of every line, the size a solve's time is estimated by */
const stopCount = (svc: Service) => svc.lines.reduce((n, l) => n + l.stops.length, 0);

/** `corridorDay`: a kind of day's factors for the corridors' longer trips and commutes (`dayResult`) */
type Result = Extract<FromDemand, { kind: "result" }> & { corridorDay?: [number, number] };
type Reply = Extract<FromDemand, { kind: QueryKind }>;
type QueryMessage = Extract<ToDemand, { kind: QueryKind }>;
type WithoutCity<T> = T extends unknown ? Omit<T, "city" | "round"> : never;
type Query = WithoutCity<QueryMessage>;
/** T-327: where a view's question goes: the solve and round the workers are asked about, the shown
 * result's version the answer is given for, and whether a worker short of the round waits for it.
 * `shown` (T-402): the round published of the rounds `round` counts (the cluster's solved ahead, or
 * the solve's), when not the shown view's */
interface Target { version: number; round: number; view: number; wait: boolean; solve?: Solve; shown?: number }
/** A question about a round in hand, answered for the view `view` */
const shownTarget = (version: number, round: number, view = version): Target => ({ version, round, view, wait: false });
/** T-4265: bubble sums' parts holding every period exactly once */
function everyPeriodOnce(parts: { mask: number }[]): boolean {
  let m = 0;
  for (const p of parts) {
    if (m & p.mask) return false;
    m |= p.mask;
  }
  return m === ALL_PERIODS;
}
/** T-4265: bubble sums' parts added into one, as `cellModes` adds them (in period order, so the bubbles are the same to
 * the bit) */
function mergeSums(parts: readonly { mask: number; sums: Float32Array; everyday: Float32Array | null }[]): { mask: number; sums: Float32Array; everyday: Float32Array | null } {
  let mask = 0, sums: Float32Array | null = null, everyday: Float32Array | null = null;
  for (const p of [...parts].sort((a, b) => a.mask - b.mask)) {
    mask |= p.mask;
    if (!sums) sums = p.sums.slice();
    else addInto(sums, p.sums);
    if (p.everyday) { if (!everyday) everyday = p.everyday.slice(); else addInto(everyday, p.everyday); }
  }
  return { mask, sums: sums!, everyday };
}
/** `into[k] += from[k]` for every k of `into` */
function addInto(into: Float32Array | Float64Array, from: ArrayLike<number>) {
  for (let k = 0; k < into.length; k++) into[k] += from[k];
}
/** A view's answer about `city` (the zone in view, unless `other`, a nearby one) still goes with the
 * solve and the kind of day shown. */
function stillShown(other: CityId | undefined, city: CityId, version: number, day: DemandDay): boolean {
  return (other !== undefined || activeCity.peek() === city) && demandView.peek()?.version === version && shownDay.peek() === day;
}
const frameOf = (): FrameSpec => { const f = getWorldFrame(); return { kind: f.kind, lon: f.lon, lat: f.lat }; };
/** T-378: the shares of the longer trips not by rail that fly and that drive on each corridor with
 * place `own`, 2 per place (`SumsCtx.modes`, game/corridorFigures.ts `ownModesFrom`); null with no
 * place, before any are known, or with no worker holding the book's market (all by car). */
const ownModesMade = new WeakMap<CorridorBook, Map<number, Float64Array>>();
async function ownModes(b: CorridorBook | null, own: number): Promise<Float64Array | null> {
  if (!b || own < 0) return null;
  const made = ownModesMade.get(b) ?? ownModesMade.set(b, new Map()).get(b)!;
  const had = made.get(own);
  if (had) return had;
  const r = await rowsNow({ ...NO_CORRIDORS, book: b }, own);
  if (!r) return null;
  const out = ownModesFrom(r, own);
  made.set(own, out);
  return out;
}
let placeCentres: Float64Array | null = null;
/** Each market place's first registry zone's centre, lon, lat (NaN: none); `SumsCtx.centres`. */
function centresOfPlaces(): Float64Array {
  if (!placeCentres) {
    placeCentres = new Float64Array(2 * INTERCITY_PLACES.length).fill(NaN);
    for (let p = 0; p < INTERCITY_PLACES.length; p++) {
      const c = cityOfPlace(p);
      if (c) (placeCentres[2 * p] = c.center[0]), (placeCentres[2 * p + 1] = c.center[1]);
    }
  }
  return placeCentres;
}
const MARKET_SPEC: FrameSpec = { kind: MARKET_FRAME.kind, lon: MARKET_FRAME.lon, lat: MARKET_FRAME.lat };

/** Workers' replies in the order of the periods they own (each worker's set is fixed), so sums and
 * fingerprints do not depend on which worker answered first. */
function byMask(parts: Reply[]): Reply[] {
  const m = (r: Reply) => ("mask" in r ? r.mask : 0);
  return [...parts].sort((a, b) => m(a) - m(b));
}
/** T-3373 B: the answers about a split solve when between them they cover every population of `need` over every
 * period, none twice, in the order of the populations they cover (so `byMask`, all masks alike, keeps one order
 * whichever worker answered first); else none, which every asker reads as not answered. So a panel never shows
 * one worker's part of a figure. `need` null (transfers): every population of the clusters holding a route node asked
 * about (`of`), so a station's transfers wait for its own cluster alone (T-392: a cluster solved ahead). */
function covered(replies: Reply[], need: string[] | null): Reply[] {
  const got = new Map<string, number>();
  need ??= [...new Set((replies as Answer[]).flatMap((r) => r.of ?? []))];
  for (const r of replies as Answer[]) {
    const m = "mask" in r ? r.mask : ALL_PERIODS;
    for (const p of r.pops ?? []) {
      const had = got.get(p) ?? 0;
      if (had & m) return [];
      got.set(p, had | m);
    }
  }
  if (!need.every((p) => got.get(p) === ALL_PERIODS)) return [];
  const key = (r: Reply) => ((r as Answer).pops ?? []).join(",");
  return [...replies].sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
}
/** `CellModes` from the workers' eight blocks a cell (demand.worker.ts `withNone`): rail, walk and
 * drive at the home end, the same at the activity end, then No route at each (T-2215). `copy`
 * slices, else the arrays view `data`. */
function cellModesOf(version: number, day: DemandDay, data: Float32Array, copy = false): CellModes {
  const n = data.length / 8, b = (k: number) => (copy ? data.slice(k * n, (k + 1) * n) : data.subarray(k * n, (k + 1) * n));
  return { version, day, homeRail: b(0), homeWalk: b(1), homeDrive: b(2), workRail: b(3), workWalk: b(4), workDrive: b(5), homeNone: b(6), workNone: b(7) };
}
const AIRPORT_KIND = EVERYDAY_KEYS.indexOf("airport");
/** T-3512, dev `?localShare=a,b`: each solving worker's share for skipping an edit's local solve (`ToDemand` `localShare`) */
const LOCAL_SHARE = import.meta.env.DEV ? new URLSearchParams(location.search).get("localShare")?.split(",").map(Number).filter((x) => x >= 0) : undefined;
/** T-3503: what the last regional answers may hold (a zone's answer is 48 bytes a square: New York about 2 MB,
 * and a zone asks for about ten kinds and days) */
const LAST_REG_BYTES = 48e6;
/** T-4277: the zones whose airports' trips across their line `lastAir` keeps (a few terminal cells each) */
const LAST_AIR_ZONES = 256;
/** T-3503: `regLegs`' kinds: longer trips (bit 0), commutes (bit 1), each everyday kind (bit 2 + k) */
const REG_COMMUTES = 2;
const REG_ALL = 3 | EVERYDAY_KEYS.reduce((b, _, k) => b | (1 << (2 + k)), 0);
/** T-3503: a zone's 600 m squares (market metres), each cell's square and its share of it at each end */
interface ZoneSq { xy: Float32Array; sq: Uint32Array; share: Float32Array }
/** A bare zone's everyday legs per solving worker's periods (T-325), in `byMask`'s order. */
function bareParts(r: Extract<FromDemand, { kind: "bare" }>): Float32Array[] {
  return r.parts.map((p, i) => [r.masks[i], p] as const).sort((a, b) => a[0] - b[0]).map(([, p]) => p);
}

function concat(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** Sort parallel arrays by value, most first, keeping at most `top`. */
function topOf(keys: ArrayLike<number>, vals: ArrayLike<number>, top = Infinity): { keys: Uint32Array; vals: Float32Array } {
  const idx: number[] = [];
  for (let i = 0; i < vals.length; i++) if (vals[i] > 0) idx.push(i);
  idx.sort((a, b) => vals[b] - vals[a]);
  if (idx.length > top) idx.length = top;
  return { keys: Uint32Array.from(idx, (i) => keys[i]), vals: Float32Array.from(idx, (i) => vals[i]) };
}

/** Each route node's stop and line: stop k of a line is node o + k on run 0 and o + 2n - 1 - k on run 1. */
function routeNodes(layout: Solve["layout"]): RouteNodes {
  const at: RouteNodes = [];
  for (const { id, o, n, stops, back } of layout) for (let k = 0; k < n; k++) {
    at[o + k] = { station: stops[k], line: id, k, o, n };
    at[o + 2 * n - 1 - k] = { station: back?.[k] ?? stops[k], line: id, k, o, n };
  }
  return at;
}
/** `members`' indices in the solve's station list, those it has */
const stationIndices = (solve: Solve, members: string[]) => members.map((m) => solve.svc.stations.findIndex((s) => s.id === m)).filter((i) => i >= 0);

/** T-2624: a local result's steps shown before all its periods were in: the views shown before the first (each
 * step starts from them), and when each step was asked and shown */
interface Step { base: DayDemand[] | null; asked: number[]; shown: number[] }

/** One solve's bookkeeping, and its timings (window.twDemand.log with ?debug=1). */
interface Solve {
  version: number;
  svc: Service;
  /** per line, the first route node and stop count; `back`: run 1's stations where they differ (T-3348) */
  layout: { id: string; o: number; n: number; stops: string[]; back?: string[] }[];
  results: (Result | null)[];
  byRound: (Result | null)[][];
  published: number;
  /** performance.now(): the network arrived, was sent to the workers, the free-flow estimate and
   * the refined result were published */
  arrivedAt: number;
  sentAt: number;
  firstAt?: number;
  refinedAt?: number;
  /** T-793: when each round was published (the loading bar's pace) */
  roundAt?: number[];
  /** T-793: when the first period of each round arrived */
  roundFirstAt?: number[];
  /** T-324: when new or changed lines got the first estimate's figures early, and which */
  earlyAt?: number;
  earlyLines?: string[];
  periodMs: number[][];
  setupMs: number[];
  /** WASM linear memory per worker after its last stage, bytes */
  memBytes: number[];
  /** the cities open in every worker when it was sent: the solve covers at least these (T-149) */
  cities: CityId[];
  /** T-396, T-377: the zones whose market packs were open in every worker when it was sent: the
   * solve covers them */
  markets: CityId[];
  /** T-1223: of `cities`, those open coarse when it was sent */
  coarse?: CityId[];
  /** T-2579: `cities` and `markets` as sent, which the cache keys its result by; a close afterwards
   * (`closeFull`) trims the two above, but the figures still hold what was closed */
  sent?: { cities: CityId[]; markets: CityId[] };
  /** T-232: its estimates before the final round are shown (startup, a game load, a city with no
   * figures yet); otherwise the last final result stays shown until this one's final round */
  showEstimates: boolean;
  /** T-2793: a final result is shown and this solve covers zones with no figures: its rounds give those zones their
   * figures alone (`fillNewZones`), and the lines stay as shown */
  newZones?: boolean;
  /** T-2793: the zones `fillNewZones` has given this solve's figures */
  filled?: Set<CityId>;
  /** T-1533: a rough solve of the zone in view alone, sent while a whole solve waits for packs; never final */
  rough?: boolean;
  /** T-3373 B: the solving worker whose `plan` names what is solved (a rough solve's holder under `?split=1`), else 0 */
  lead?: number;
  /** T-3373 B: the holding plan it was sent with (`?split=1`), which its questions go by */
  plan?: Omit<SplitPlan, "me">;
  /** T-326: the last worker's clusters kept and solved */
  clusters?: [number, number];
  /** T-124: per solving worker, its periods kept and solved in the clusters it solved */
  periods?: [number, number][];
  /** T-1609: the zones the last worker solved alone against their clusters' frozen loads */
  local?: CityId[];
  /** T-1733: per solving worker, how many clusters it solves locally first (`plan.local`), and
   * whether `whole` went */
  localPlans?: number[];
  /** T-2729: per solving worker, an edit's solve's last `work` */
  work?: WorkPart[];
  /** T-3599, ?debug=1: every `work` as it came, by worker */
  workLog?: (WorkPart & { i: number })[];
  /** T-3599: when the last unit of an edit's solve ended, on any worker; its local stage's last unit
   * ended; its whole solve's first unit began */
  lastUnitAt?: number;
  localEndAt?: number;
  wholeAt?: number;
  wholeSent?: boolean;
  /** T-2188: per solving worker, whether its local stage is a city edit's (`plan.final`); whether `endLocal` went, so
   * the local results end the solve; per worker, whether it holds a local result no whole solve has replaced since
   * (`done.localHeld`) */
  ends?: boolean[];
  endLocal?: boolean;
  localHeld?: boolean[];
  /** T-2188: the idle refresh, a whole solve of the network shown: quiet (no status, no clock cap) until an edit
   * waits on it (`shown`); the zones its lead worker solves (`plan.solving`), for the status once shown */
  refresh?: { shown: boolean; solving?: CityId[] };
  /** T-4265: the zones its lead worker's plan solves: no worker holds their figures on screen past their clusters' set-up */
  solving?: Set<CityId>;
  /** T-1790: local cluster results shown so far, and (T-1734) when the first was */
  localShown?: number;
  localShownAt?: number;
  /** solved with the round barrier (estimates by round); else after an edit, counted by stage */
  barrier?: boolean;
  /** T-4220: its solving workers were asked for the figures on screen (`snap`), which a side job is planned against */
  snap?: boolean;
  /** T-1734: networks held back while it ran, and when the first came */
  heldFrom?: number;
  /** T-3616: when the first of the edits' solves this one follows on from without a whole solve started between
   * was sent (its own `sentAt` if none); and whether a check at the end of `LOCAL_CHAIN_MS` is set */
  chainFrom?: number;
  chainTimer?: boolean;
  /** T-3925: when the first of the edits' solves this one follows on from without a final between was sent (its own
   * `sentAt` if none); and, on a solve a newer network stopped (`restarts`), how long it had run, ms */
  restartFrom?: number;
  cutAfter?: number;
  /** T-392: each cluster published ahead of the solve's final, by its cities: its results per
   * round and period as they come in, the last round in from every worker, and when its final
   * round was shown */
  clusterRes?: Map<string, { byRound: (Result | null)[][]; published: number; at?: number; roundAt?: number[]; steps?: Step }>;
  /** T-402: the cluster in view, stepped through its rounds alone ahead of the rest (a solve with
   * the barrier, `plan.ahead`): its cities and key in `clusterRes` */
  ahead?: { cities: CityId[]; key: string };
  /** T-2459: sent while the focus pool solved the zone in view, so the city view's zone's cluster
   * goes ahead of the zone in view's, whose focus figures then hold the clock's release */
  focused?: boolean;
  /** T-1543: the parts of a period's result in so far, by cluster key, round and period */
  partsIn?: Map<string, Result[]>;
  /** T-2045: a focus solve's lines with figures (`result.onlyLines`) */
  onlyLines?: Uint32Array;
}

/** When one item (a city or the market) was asked for, downloaded and open in every worker:
 * performance.now() (window.twDemand.start.loads with ?debug=1). */
interface LoadLog { item: string; requestedAt: number; downloadedAt?: number; openedAt?: number; bare?: boolean; closedAt?: number }
/** T-4220 (?debug=1, `twDemand.client.asideLog`): a side job decided. `version` its own, `on` the running whole solve's,
 * `held` whether its network was held behind that solve (else that solve's own); `at` and `endAt` performance.now() */
interface AsideLog {
  version: number; on: number; gen: number; held: boolean; at: number; endAt?: number;
  refused?: string; why?: string; superseded?: boolean; stale?: boolean; done?: boolean;
  fresh?: CityId[]; periods?: number[]; results?: number; sums?: CityId[];
  /** stage 7: its clusters put on screen, or why none was */
  shown?: number; notShown?: string;
  /** expected, ms: the plan's solve, its zones' opening, the main pool's to the edit's lines; then as they took */
  expMs?: number | null; openExp?: number; mainMs?: number | null; planMs?: number; openMs?: number; workerMs?: number;
}
/** T-4220: one cluster of a side job's figures on screen (`sideOver`): its zones, the fresh ones, its lines, those a fresh
 * zone's population reaches (`seen`, as its view on the network takes them), and its views of every kind of day */
interface SideCluster { zones: CityId[]; fresh: CityId[]; all: Solve["layout"]; seen: Set<string>; ests: DayDemand[] }
/** One file being downloaded, for the loading percentage (T-149). */
interface Download { got: number; total: number; at: number; end?: number }
/**
 * The model behind the loading bar's stages (T-243, T-1195, game/loadBar.ts) when the slot's last
 * load is not remembered, measured on Anita's desktop (notes/T-243.md, T-793, T-889): opening a city
 * pack 450 ms plus 220 ms per MB of its .bin, the market packs 20 ms per MB and one composition for them all,
 * scaled by how this browser's last load compared (`loadPace`); a download's length is its own rate
 * so far; a game load's solve from SOLVE_FIT, scaled by this browser's last game solve against it
 * (T-1479); any other load's first estimate 1 s plus 3.5 ms per line stop, scaled by this browser's
 * last solve of the zone in view (LAST_SOLVE). The solve re-measures itself round by round (`solveStage`).
 */
const OPEN_MS_PER_MB = { city: 220, market: 20 };
/** T-2552: a game solve shorter than this says nothing of this browser's pace on a big one, nor of its later rounds
 * against its first (a new game's empty network: 4.5 s against the model's 34 s, which then put an 81-zone save's
 * solve at a third of its time) */
const PACE_SOLVE_MS = 10000;
/** a solve's setup in its workers (set_network, before its first period), the longest of those reported */
const setupOf = (s: Solve) => s.setupMs.reduce((m, x) => Math.max(m, x ?? 0), 0);
/** T-3876: a solve's time left by its workers' units (`editLeft`, as the status tag's hover has it), or null while
 * a worker has not said, or can't tell yet; the load model's estimate stands in then */
function unitsLeft(s: Solve, n: number, now: number, part: "all" | "view" = "all"): number | null {
  return unitsBound(s, n, now, part)?.ms ?? null;
}
/** T-4264: the same, and whether the units bound it (`editLeftOf`): only then may a bar pass half on it */
function unitsBound(s: Solve, n: number, now: number, part: "all" | "view" = "all"): { ms: number; bounded: boolean } | null {
  const w = s.work;
  if (!w || n < 1) return null;
  for (let k = 0; k < n; k++) if (!w[k]) return null;
  return editLeftOf(w.slice(0, n), now, editPrior, part);
}
/** T-1195: and each city pack 450 ms besides its size (China's 169 packs took three times what their
 * MB alone gave, New York's 5 and the world save's 21 matched it) */
const OPEN_CITY_MS = 450;
/** composing the market from the packs opened together, nearly the same for one pack as for forty
 * (0.7 s for one of Japan's, 1.5 s for 45, 1.4-3.6 s for the world save's 22-39; T-889), shared out
 * among a load's market packs */
const OPEN_MARKET_MS = 1500;
/** T-484: the waits before each new try of a failed pack download, and, once those are spent, before
 * the zone is tried again (the last repeating) */
const RETRY_MS = [500, 1500, 4000];
const BROKEN_RETRY_MS = [10000, 30000, 60000];
/** T-1174: workers replaced after a panic in one page, before a further panic stops the pool */
const MAX_RESPAWNS = 4;
/** T-2704: the waits before each new try of a worker whose code did not download (as wasmInit.ts's) */
const SCRIPT_WAITS_MS = [1_000, 4_000];
/** T-1205 (measured, notes/T-1205.md): a fresh worker's WASM after a solve, as its heap at rest or that
 * solve's heap peak times these, plus MEM_BASE (T-3604, notes/T-3604.md: all-China within 3%, New York set by
 * its peak) */
const MEM_NEED: [number, number] = [1.12, 1.07];
const MEM_BASE = 64 * 2 ** 20;
/** T-3604: pack downloads a pool's replacement workers have under way at once, all of them together */
const REOPEN_FETCHES = 6;
/** an idle worker restarts when its WASM is this many times its need and this much above it */
const MEM_OVER = 1.15;
const MEM_FREE = (Number(new URLSearchParams(location.search).get("memFree")) || 384) * 2 ** 20; // ?memFree=MB: checks
/** and the network has been still this long; a pool busy then is looked at again after MEM_RETRY_MS */
const MEM_QUIET_MS = 20_000;
const MEM_RETRY_MS = 5_000;
/** where the browser cannot measure the page: Chrome outside the demand workers, and each demand
 * worker outside its WASM (its JS heap; about 0.7 GB a worker on the all-China save) */
const MEM_OTHER = 0.6 * 2 ** 30;
const MEM_WORKER = 0.15 * 2 ** 30; // (T-3472: 110-150 MB a solving worker on all-China)
/** T-3373: the share of the game's budget (`memBudget`) the solving workers may take together. Each holds
 * every open zone and the market, so a big save runs fewer: on a reported 8 GB, New York 2, all of China 1.
 * Not with "Use more cores" on (`shareCapped`): the player chose speed */
const SOLVING_SHARE = 0.175;
/** T-3373 B (`?split=1`): the solving workers' share once each zone and the market have one holder (Anita, 11-21: the
 * limit rises with B; 2 GB on a reported 8 GB), and what a further solving worker then adds: its WASM floor and room
 * grown ahead (all-China: 33 MB more WASM at 2 workers than at 1, notes/T-3373-B-plan.md stage 4), and its JS (`MEM_WORKER`) */
const SPLIT_SHARE = 0.25;
const SPLIT_EXTRA = 0.2 * 2 ** 30;
/** T-1366: the pool grows back (`growTo`) only to a count whose workers fit GROW_ROOM of `capPool`'s limits, and
 * no sooner than GROW_GAP_MS after its last change (doubled by each cut after a growth), so zones opening and
 * closing near the line never start and stop workers in turn */
const GROW_ROOM = 0.95;
const GROW_GAP_MS = 120_000;
/** T-4343: a period may run this many times the longest one timed on the page before the card counts it stuck (a
 * load's periods differ: one at twice the others' time on that laptop), and this long when none has ended yet */
const UNIT_QUIET_SLACK = 4;
const UNIT_QUIET_FIRST_MS = 30 * 60_000;
const mb = (bytes: number) => Math.round(bytes / 2 ** 20);
const sameBytes = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i]);
const SOLVE_MS = { base: 1000, perStop: 3.5 };
/** T-1479 (T-890): a game load's solve stage, from its network reaching demand's packs open to the card
 * going: fitted on the real New York save (5 zones, 1,756 stops, 59 s), all-China (171, 2,363, 103 s)
 * and Japan (45, 10,320, 880 s), notes/T-1479.md: Japan's tenfold time for four times New York's stops
 * takes the square of the stops. */
const SOLVE_FIT = { base: 34_000, perZone: 148, perStop2: 0.00788 };
const solveModelMs = (stops: number, zones: number) => SOLVE_FIT.base + SOLVE_FIT.perZone * zones + SOLVE_FIT.perStop2 * stops * stops;
/** a pack whose size is not known yet (its download has not started): about a mid-sized city */
const GUESS_BYTES = 6e6;
const LAST_SOLVE = "anitabuilder-load-solve";
/** T-377: a zone's market pack as a load item, "market:<zone>" */
const MARKET = "market:";
const marketItem = (id: CityId) => MARKET + id;
/** T-776: the world market's rows, beside the registry index the app bundles (a hashed file in a build) */
const MARKET_ROWS = new URL("../game/market-rows.json", import.meta.url).href;
/** T-4199: the market's corridors and corridor book, worked out once (tools/corridor_file.mjs); a build without it works them out */
const MARKET_CORRIDORS = new URL("../game/market-corridors.bin", import.meta.url).href;
/** the corridor file's bytes, null where there is none (any other answer is the worker's to refuse) */
const CORRIDOR_RETRY_MS = [500, 1500, 4000];
const fetchCorridorFile = async (): Promise<Uint8Array | null> => {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await patientFetch(MARKET_CORRIDORS);
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`corridor file: ${r.status}`);
      return new Uint8Array(await r.arrayBuffer());
    } catch {
      // (a file that never comes is no failure: the corridors are worked out, as without one)
      if (attempt >= CORRIDOR_RETRY_MS.length) return null;
      await new Promise((res) => setTimeout(res, CORRIDOR_RETRY_MS[attempt]));
    }
  }
};
const isMarket = (item: string) => item.startsWith(MARKET);
const marketZone = (item: string) => item.slice(MARKET.length) as CityId;
/**
 * T-301: a pack no longer needed (not viewed, beside the view, reached by a station, or part of the
 * load in hand; a market pack: its zone not so, and no station in its reach) closes in every worker
 * once it has gone unneeded for CLOSE_MS, or at once if more than KEEP_SPARE unneeded zones are open
 * (the least recently needed close first). It reopens when needed, as on first opening.
 */
const CLOSE_MS = 60_000;
const KEEP_SPARE = 2;
/** the newest load matching `f` (a pack can be opened, closed and opened again) */
const lastLoad = (loads: LoadLog[], f: (l: LoadLog) => boolean) => { for (let i = loads.length - 1; i >= 0; i--) if (f(loads[i])) return loads[i]; return undefined; };

const solveModel = (stops: number) => SOLVE_MS.base + SOLVE_MS.perStop * stops;
/** T-793: the periods each solving worker takes in turn (5 periods on 3 workers: 2); SOLVE_MS was
 * measured with one each, so a game with no remembered solve expects that many times as long */
let solveWaves = 1;
/** T-347: until a load has timed its final round, the final is guessed at this many first estimates */
const FINAL_FACTOR = 1 + ROUNDS * LATER_GUESS;
/** `finalMs`: to the card going (T-421: the zone in view's final); `allMs`: to the solve's final */
interface SolveRecord { ms: number; stops: number; finalMs?: number; allMs?: number }
/** T-793: the record of the zone in view, else the last one made: a Tokyo solve takes three times a New
 * York one of the same stops, so games in different zones do not share one. */
function lastSolve(): SolveRecord | null {
  try {
    const v = JSON.parse(localStorage.getItem(LAST_SOLVE) || "null");
    const r = v?.by?.[activeCity.peek()] ?? v;
    return r && r.ms > 0 && r.stops >= 0 ? r : null;
  } catch { return null; }
}
function rememberSolve(ms: number, stops: number, finalMs?: number, allMs?: number) {
  // (T-884: an empty game's solve, the market alone, would stand in for every zone with no record)
  if (stops <= 0) return;
  try {
    const rec: SolveRecord = { ms, stops, ...(finalMs ? { finalMs } : {}), ...(allMs ? { allMs } : {}) };
    const city = activeCity.peek();
    const by: Record<string, SolveRecord> = {};
    for (const [k, v] of Object.entries(JSON.parse(localStorage.getItem(LAST_SOLVE) || "null")?.by ?? {}).slice(-7)) if (k !== city) by[k] = v as SolveRecord;
    by[city] = rec;
    localStorage.setItem(LAST_SOLVE, JSON.stringify({ ...rec, by }));
  } catch { /* storage unavailable */ }
}
/** T-793: the stops a game is guessed to have until its network arrives, with no solve remembered */
const GUESS_STOPS = 1000;
/** Expected ms from a solve's start to its first estimate, or (`toFinal`, T-347) to the card going,
 * or ("all") to the solve's final; `stops` null = the network is not known yet. */
function solveEstimate(stops: number | null, toFinal: boolean | "all" = false): number {
  const last = lastSolve();
  const n = stops ?? last?.stops ?? GUESS_STOPS;
  const final = toFinal === "all" ? last?.allMs ?? last?.finalMs : last?.finalMs;
  const ms = last ? (toFinal ? final ?? last.ms * FINAL_FACTOR : last.ms) * solveModel(n) / solveModel(last.stops) : solveModel(n) * solveWaves * (toFinal ? FINAL_FACTOR : 1);
  return ms;
}

/** T-2410: a station no line stops at changes no figure, so building stations ahead of their line
 * starts no solve (it would set every population on its network, and the line after it could not be
 * solved locally) */
function serviceFromSnapshot(s: NetworkSnapshot): Service {
  const used = new Set(perfOff("alone") ? s.stations.map((st) => String(st.id)) : s.lines.flatMap((l) => [...l.stops, ...(l.backStops ?? [])].map(String)));
  const alone = s.stations.filter((st) => !used.has(String(st.id))).map((st): [number, number] => [st.x, st.y]);
  const passages = (s.passages ?? []).filter(([a, b]) => used.has(String(a)) && used.has(String(b)));
  const apart = (s.apart ?? []).filter(([a, b]) => used.has(String(a)) && used.has(String(b)));
  return {
    frame: s.frame ?? LEGACY_FRAME,
    fare: s.fare ?? { base: 2, perKm: 0 }, // legacy fixed 6 perceived minutes
    stations: s.stations.filter((st) => used.has(String(st.id))).map((st) => ({ id: String(st.id), x: st.x, y: st.y, complex: st.complex, ...(st.level ? { level: st.level } : {}), ...(st.upgrades ? { upgrades: st.upgrades } : {}), ...(st.fare ? { fare: st.fare } : {}), ...(st.street ? { street: true } : {}) })),
    lines: s.lines.map((l) => ({ id: String(l.id), stops: l.stops.map(String), ...(l.backStops?.length === l.stops.length ? { back: l.backStops.map(String) } : {}), tph: l.tph, hopS: l.hopS, hopKm: l.hopKm, ...(l.fast ? { fast: true } : {}), ...(l.ferry ? { ferry: true } : {}), ...(l.boat !== undefined ? { boat: l.boat } : {}), ...(l.brt ? { brt: true } : {}), ...(l.tram ? { tram: true } : {}), ...(l.fareDist ? { fareDist: true } : {}), ...(l.passFares ? { passFares: l.passFares } : {}), dwellS: l.dwellS, ...(l.wrapS ? { wrapS: l.wrapS } : {}), cars: l.cars, ...(l.span ?{ span: l.span } : {}), ...(l.catches ? { catches: l.catches } : {}) })),
    ...(passages.length ? { passages: passages.map(([a, b]) => [String(a), String(b)] as [string, string]) } : {}),
    ...(apart.length ? { apart: apart.map(([a, b]) => [String(a), String(b)] as [string, string]) } : {}),
    ...(alone.length ? { alone } : {}),
  };
}

/** T-1733: after an edit's local results are shown, how long the whole solve waits for the next edit
 * (stations built, then the line, a few seconds apart): until it starts, the next edit is solved locally too */
const EDIT_SETTLE_MS = 8000;
/** T-2188: a city edit's solve ends at its local result, so figures drift a little edit by edit (the untouched cities
 * and the market keep their loads): this long after the last edit, with a city edit's result shown and no whole solve
 * of it since, the network is solved whole again, quietly (`refresh`). Its whole solve takes 3-5 minutes on a big
 * connected save, and an edit made during it waits as for any whole solve (T-3925), so it waits for a real pause in
 * building rather than the next few seconds (notes/T-2188.md) */
const REFRESH_IDLE_MS = 120_000;
/** T-3616: an edit made during an edit's local solve goes as soon as that local result shows, while the first
 * edit since a whole solve last started was sent at most this long before; after that it waits for the whole
 * solve's final (T-1734), so a local solve slower than this (all of Japan's, about 2 minutes) never restarts */
const LOCAL_CHAIN_MS = 60000;
/** T-3616: for this long after an edit's solve is sent, a network held behind its local solve is taken to go at
 * the local result: no zone is solved aside (T-2605) and the time left counts no second solve (T-3599); a slow
 * local solve gets both after it, so all of Japan's aside starts at most this much later */
const HOLD_QUIET_MS = 10000;
/** T-3925: a newer network stops an edit's whole solve and starts again while under this share of it has run (by
 * the time left, `unitsLeft`), and the edits solved one after another without a final began less than one such solve
 * ago, so building steadily still lands a final within about two solves of the first edit (T-1734). Past about 40%
 * the held edit's local result on top of the running final shows its lines sooner than a restarted whole solve,
 * which has no local stage (on the NA save, notes/T-3925.md) */
const RESTART_SHARE = 0.4;
/** T-4220 (notes/T-4220.md, "The pick" 5): an edit made during a whole solve is solved aside this long after the last
 * edit, so edits arriving together share one side job */
const ASIDE_QUIET_MS = 1000;
/** T-4220: a side job starts only where its expected time is under this share of the main pool's to the edit's lines
 * (first guess) */
const ASIDE_GAIN = 0.7;
/** T-4220: the side pool's start, before its plan can be asked for (0.5 s on New York, stage 3; a guess for slower machines) */
const ASIDE_START_MS = 1500;
/** T-4220: the side pool's failures in one game after which no side job starts until another game */
const ASIDE_FAILS = 2;

/** Pack a service for the workers; lines naming a station that is not in the list are left out. */
function packService(svc: Service): { net: DemandNetwork; layout: Solve["layout"]; ids: { lines: string[]; stations: string[] } } {
  const index = new Map(svc.stations.map((s, i) => [s.id, i]));
  const street = new Set(svc.stations.filter((s) => s.street).map((s) => s.id));
  const lines = svc.lines.filter((l) => l.stops.length >= 2 && l.stops.every((s) => index.has(s)));
  const nStops = lines.reduce((a, l) => a + l.stops.length, 0);
  const net: DemandNetwork = {
    frame: svc.frame,
    stXY: new Float32Array(svc.stations.flatMap((s) => [s.x, s.y])),
    // a station without a complex number is one of its own: numbered past every complex
    complex: new Uint32Array(svc.stations.map((s, i) => s.complex ?? svc.stations.length + i)),
    ...(svc.stations.some((s) => s.upgrades) ? { ups: Uint8Array.from(svc.stations, (s) => s.upgrades ?? 0) } : {}),
    // T-3740: the stations' fare zones, only where the fares have areas and a station has one
    ...(svc.fare.zones?.length && svc.stations.some((s) => s.fare) ? { fz: Uint32Array.from(svc.stations, (s) => s.fare ?? 0) } : {}),
    ...(svc.passages?.length ? { passages: Uint32Array.from(svc.passages.filter(([p, q]) => index.has(p) && index.has(q)).flatMap(([p, q]) => [index.get(p)!, index.get(q)!])) } : {}),
    ...(svc.apart?.length ? { apart: Uint32Array.from(svc.apart.filter(([p, q]) => index.has(p) && index.has(q)).flatMap(([p, q]) => [index.get(p)!, index.get(q)!])) } : {}),
    lineN: new Uint32Array(lines.map((l) => l.stops.length)),
    stops: new Uint32Array(nStops),
    times: new Float32Array(TIMES_PER_STOP * nStops),
    tph: new Float32Array(lines.flatMap((l) => l.tph)),
    cars: new Float32Array(lines.map((l) => l.cars)),
    fare: svc.fare,
    hopKm: new Float32Array(2 * nStops),
    span: lines.some((l) => l.span) ? new Float32Array(PERIODS.length * 2 * nStops).fill(1) : new Float32Array(0),
    // T-2196: the ferry fans' lines, only where there are any (`?perfOff=fans`: no fans, for timing)
    ...(lines.some((l) => l.ferry) && !perfOff("fans") ? { ferry: Uint8Array.from(lines, (l) => (l.ferry ? 1 : 0)) } : {}),
    // T-2926: tram lines, only where there are any (`DemandApi.set_tram_lines`)
    ...(lines.some((l) => l.tram) ? { tram: Uint8Array.from(lines, (l) => (l.tram ? 1 : 0)) } : {}),
    // T-3888: the boarding cost (`set_stairs`): ferry lines never pay, nor trams at a stop on the street (T-4041)
    ...(lines.some((l) => l.ferry) ? { pierLines: Uint8Array.from(lines, (l) => (l.ferry ? 1 : 0)) } : {}),
    ...(street.size ? { streetStops: Uint8Array.from(lines.flatMap((l) => l.stops.map((s) => (street.has(s) ? 1 : 0)))) } : {}),
  };
  // T-3348: run 1's stations, only where a line has a platform beside for its way back
  const backOf = (l: (typeof lines)[number]) => (l.back?.length === l.stops.length && l.back.every((s) => index.has(s)) ? l.back : null);
  if (lines.some((l) => backOf(l))) net.back = new Uint32Array(nStops);
  // T-1626, T-2195: a high-speed, ferry or bus line's route nodes charge its multiplier, only where one is not 1;
  // T-3567: times the line's own
  const perNode = (muls: number[]) => {
    const w = new Float32Array(2 * nStops).fill(1);
    let o = 0;
    lines.forEach((l, i) => {
      if (muls[i] !== 1) w.fill(muls[i], 2 * o, 2 * (o + l.stops.length));
      o += l.stops.length;
    });
    return w;
  };
  const muls = lines.map((l) => lineFareMul({ id: l.id, fast: l.fast, ferry: l.ferry, boat: l.boat, bus: !!l.catches && !l.ferry, brt: l.brt }, svc.fare));
  if (muls.some((m) => m !== 1)) net.fareW = perNode(muls);
  // T-3578: the line's own multiple scales the base too, a trip paying it times the largest of its lines'
  const own = lines.map((l) => ownLineFare(svc.fare, l.id));
  if (own.some((m) => m !== 1)) net.baseW = perNode(own);
  // T-3748, T-3753: every trip's extras are charged at the scale of the stop each route node leaves
  const w = world.peek();
  if (w) {
    const byId = stopsById(w), fareS = new Float32Array(2 * nStops).fill(1), baseS = new Float32Array(2 * nStops).fill(1);
    let o = 0;
    for (const l of lines) {
      const n = l.stops.length, at = l.stops.map((id) => stopFareScales(byId.get(id), svc.fare));
      // run 1's k-th route node leaves stop n - 1 - k
      for (let k = 0; k < n; k++) {
        fareS[2 * o + k] = at[k].rate; fareS[2 * o + n + k] = at[n - 1 - k].rate;
        baseS[2 * o + k] = at[k].base; baseS[2 * o + n + k] = at[n - 1 - k].base;
      }
      o += n;
    }
    if (fareS.some((v) => v !== 1)) net.fareS = fareS;
    if (baseS.some((v) => v !== 1)) net.baseS = baseS;
  }
  const span = net.span!;
  const passAt = new Map<number, number[]>();
  const layout: Solve["layout"] = [];
  let a = 0;
  for (const l of lines) {
    const n = l.stops.length;
    const back = net.back ? backOf(l) : null;
    layout.push({ id: l.id, o: 2 * a, n, stops: l.stops, ...(back ? { back } : {}) });
    l.stops.forEach((s, k) => (net.stops[a + k] = index.get(s)!));
    if (net.back) (back ?? l.stops).forEach((s, k) => (net.back![a + k] = index.get(s)!));
    // T-3542: per demand period
    for (let q = 0; q < PERIODS.length; q++) {
      const at = (run: number) => TIMES_PER_STOP * a + (run * PERIODS.length + q) * n;
      const h0 = l.hopS[0]?.[q] ?? [], d0 = l.dwellS[0]?.[q] ?? [];
      const h1 = l.hopS[1]?.[q] ?? [], d1 = l.dwellS[1]?.[q] ?? [];
      for (let k = 0; k < n - 1; k++) net.times[at(0) + k] = (h0[k] ?? 0) + (d0[k] ?? 0);
      // run 1 travels the stops reversed: its j-th hop leaves stop n - 1 - j
      for (let j = 0; j < n - 1; j++) net.times[at(1) + n - 1 - j] = (h1[j] ?? 0) + (d1[j] ?? 0);
      // T-798: a circle rides on round from its last stop to its first (`demand::Network::line_wrap`)
      if (l.wrapS) net.times[at(0) + n - 1] = net.times[at(1)] = l.wrapS;
    }
    // Fare distances use actual track length. Old/test snapshots fall back to station distance.
    for (let run = 0; run < 2; run++) for (let k = 0; k < n - 1; k++) {
      const i = run === 0 ? k : n - 1 - k, j = run === 0 ? k + 1 : n - 2 - k;
      const from = svc.stations[index.get(l.stops[i])!], to = svc.stations[index.get(l.stops[j])!];
      net.hopKm[2 * a + run * n + k] = l.hopKm?.[run]?.[k] ?? Math.hypot(to.x - from.x, to.y - from.y) / 1000;
    }
    // T-3758: the stations each route node's trains pass to the next stop, as `hopKm`
    if (net.fz && l.passFares) for (let run = 0; run < 2; run++) for (let k = 0; k < n - 1; k++) {
      const c = l.passFares[run]?.[k];
      if (c?.length) passAt.set(2 * a + run * n + k, c);
    }
    if (l.span?.length === PERIODS.length * 2 * n) {
      for (let q = 0; q < PERIODS.length; q++) for (let r = 0; r < 2 * n; r++) span[q * 2 * nStops + 2 * a + r] = l.span[q * 2 * n + r];
    }
    a += n;
  }
  // T-1464: each stop's stretch of bus route, parallel to the stops
  if (lines.some((l) => l.catches)) {
    const pts: number[] = [];
    net.catchN = new Uint32Array(nStops);
    let k = 0;
    for (const l of lines) {
      for (let j = 0; j < l.stops.length; j++, k++) {
        const c = l.catches?.[j] ?? [];
        net.catchN[k] = c.length / 2;
        for (const v of c) pts.push(v);
      }
    }
    net.catchXY = new Float32Array(pts);
  }
  // T-3758: lines paying the distance fare and the stations passed, only with fare zones
  if (net.fz) {
    if (lines.some((l) => l.fareDist)) net.fzDist = Uint8Array.from(perNode(lines.map((l) => (l.fareDist ? 0 : 1))), (v) => (v === 0 ? 1 : 0));
    if (passAt.size) {
      const off = new Uint32Array(2 * nStops + 1), codes: number[] = [];
      for (let r = 0; r < 2 * nStops; r++) { for (const c of passAt.get(r) ?? []) codes.push(c); off[r + 1] = codes.length; }
      net.fzPassOff = off;
      net.fzPass = Uint32Array.from(codes);
    }
  }
  return { net, layout, ids: { lines: layout.map((l) => l.id), stations: svc.stations.map((s) => s.id) } };
}

class DemandClient {
  /** The pool's workers that solve periods, then one more with none (`aux`, demand.worker.ts), except on a phone. */
  private workers: Worker[] = [];
  private computeN = 0;
  /** T-4103: a phone's pool of one solving worker has no aux worker: worker 0 holds the zones open bare and answers
   * the aux worker's questions between its populations (65-80 MB less; never with `?split=1`) */
  private auxShared = false;
  /** the worker that holds the zones open bare and answers the aux worker's questions */
  private get auxAt(): number { return this.auxShared ? 0 : this.computeN; }
  private periodsOf: number[][] = [];
  /** T-1607: the periods whose market each solving worker solves in a solve with the barrier */
  private marketsOf: number[][] = [];
  private marketsMsg(i: number) { const m = this.marketsOf[i] ?? []; return { markets: m, share: this.marketsOf.some((qs, j) => qs.join() !== this.periodsOf[j]?.join()) }; }
  /** each worker's periods for `n` solving workers (the aux worker none). T-3373 B (`?split=1`): every solving
   * worker solves every period of what it holds, the market with its cities (no T-1607 sharing) */
  private dealPeriods(n: number) {
    const all = PERIODS.map((_, q) => q);
    this.periodsOf = [...Array.from({ length: n }, (_, i) => (this.holding ? all : periodsOfWorker(i, n))), []];
    this.marketsOf = this.holding ? this.periodsOf.slice(0, n) : marketsOfWorkers(n);
  }
  /** the aux worker's groups: each solving worker's periods (T-325); under the split one group of all */
  private auxGroups(): number[][] {
    return this.holding ? [PERIODS.map((_, q) => q)] : this.periodsOf.slice(0, this.computeN);
  }

  // --- T-3373 B stage 3: who holds what (`?split=1`; `?split=all` and off: every worker) ---
  /** each unit's one solving worker under `?split=1`; null otherwise */
  private holding: Holding | null = SPLIT === "plan" ? new Holding(1) : null;
  /** each city pack's cells' box (from a worker's `opened`), which workers not holding it join clusters by */
  private boxes = new Map<CityId, Box>();
  /** each city pack's everyday purposes (from a solving worker's `opened`; the aux worker opens none) */
  private kinds = new Map<CityId, number>();
  /** the solving workers holding `item` (a zone, or "market:<id>"), placing it if it has no holder yet */
  private solvingHolders(item: string): number[] {
    if (!this.holding) return Array.from({ length: this.computeN }, (_, i) => i);
    const unit = isMarket(item) ? MARKET_UNIT : item;
    return [this.holding.holder(unit, isMarket(item) ? 1 : cityById(item as CityId)?.localTrips ?? 1)];
  }
  /** the workers an item is posted to and must be acknowledged by: its solving holders and the aux worker (a
   * market pack's as empty bytes) */
  private holdersOf(item: string): number[] {
    return [...new Set([...this.solvingHolders(item), this.auxAt])];
  }
  private holds(i: number, item: string): boolean {
    return i === this.auxAt || this.solvingHolders(item).includes(i);
  }
  /** every worker's bubble sums for a round of the newest solve, sent unasked (T-119): `version:round:city:day`; and
   * (T-4265) each zone's of the final it shows, until a newer final of it (`keepShownSums`) */
  private pushed = new Map<string, { mask: number; sums: Float32Array; everyday: Float32Array | null }[]>();
  private ready = 0;
  /** demand_periods() from the first worker: PERIOD_STRIDE numbers per period */
  private periodTable: number[] = [];
  private version = 0;
  private cur: Solve | null = null;
  /** T-392: zones whose shown figures come from a cluster published ahead of a newer solve's final
   * (the version they came from); questions about them go to that solve */
  private cityVersion = new Map<CityId, number>();
  /** T-378: the corridor book last sent, whose rows hold the shares of the longer trips not by rail
   * by plane and by car (`ownModes`) */
  private modesBook: CorridorBook | null = null;
  /** T-2136: the rows asked of the solving workers (`corridorRows`), by question */
  private rowsAsked = new Map<number, { left: number; done: (r: CorridorRows | null) => void }>();
  private rowsNext = 1;
  /** T-1208, T-1210: the sums worker (`inSums`), started on first use, and its questions waiting */
  private sumsWorker: Worker | null = null;
  private sumsNext = 1;
  private sumsWaiting = new Map<number, (v: unknown) => void>();
  private sumsBroken = false;
  /** sums worker: the cells it holds the curve ranks of (`flowsAt`), oldest first */
  private sumsCells = new Set<string>();
  /** T-1208: the publishes waiting for their views (`inOrder`) */
  private dayQueue: Promise<void> = Promise.resolve();
  /** T-1008: the market's corridor books by key, the newest last (`dayCorridors`) */
  private corridorBooks = new Map<string, CorridorBook>();
  /** the books asked for again (`corridorsAgain`) and not back yet */
  private tableAsked = new Set<string>();
  /** T-789: the demand cache's writer (`writeStored`) and its writes waiting for an answer */
  private cacheWriter: Worker | null = null;
  private cacheWriteWaits = new Map<number, (ok: boolean) => void>();
  private cacheWriteId = 0;
  /** a service waiting for the pool to start */
  private waiting: { svc: Service; arrivedAt: number } | null = null;
  private lastKey = "";
  /** T-2409: the edits' networks no final has covered yet, oldest first, with the game minute each came */
  private uncovered: { svc: Service; minute: number }[] = [];
  readonly log: Solve[] = [];
  private nextQuery = 1;
  private pendingQueries = new Map<number, { left: number; replies: Reply[]; done: (r: Reply[]) => void; focus?: boolean }>();
  /** T-285: the last queries' timings (epoch ms): sent, and per reply its worker, when the worker
   * took it up, how long it took, when it landed here (window.twDemand.client.qlog, ?debug=1) */
  qlog: { id: number; kind: string; sent: number; replies: { w: number; start: number; ms: number; at: number }[] }[] = [];

  /** T-1165: a bubble selection's far end being worked out; its longer trips wait for it, so the
   * market's selection sum (1-3 s on New York's real save at 2 cores) does not hold it up */
  private flowsPending: Promise<unknown> | null = null;
  /** T-2746: settles once that far end's `flows` question is posted to worker 0 (or it ends without
   * one): a longer-trips question posted before it would hold it up in the worker's queue */
  private flowsPosted: Promise<void> | null = null;
  private focusEpoch = 0;
  private focusCity = activeCity.peek();
  private focusFrame = worldFrameEpoch.peek();
  private cellsP: Promise<CityCells | null> | null = null;
  private modesP: { version: number; round: number; day: DemandDay; src?: number; p: Promise<CellModes | null> } | null = null;
  /** T-260 `day_factors()` from the first worker: load factor per kind of day, purpose and
   * period, then each day's trip rate per purpose (sim/src/demand/days.rs) */
  private dayTable: number[] = [];
  /** `purposeModes` answers: `version:round:purpose` */
  private purposeP = new Map<string, Promise<CellModes | null>>();
  /** T-327: view answers asked of the running solve by round, `question|version|round` */
  private aheadP = new Map<string, { version: number; round: number; p: Promise<unknown> }>();
  readonly start: { packMs?: number; postMs?: number; workers: { wasmMs: number; openMs: number; warmMs: number; memBytes: number }[]; loads: LoadLog[] } = { workers: [], loads: [] };
  status = "loading";
  /** the pool has stopped (`fail`) */
  private get failed() { return this.status.startsWith("failed"); }
  // --- what is loaded (T-149) ---
  /** city packs and the market ("market") being fetched or fetched */
  private fetching = new Set<string>();
  private downloads = new Map<string, Download>();
  /** workers that have opened each item, and the items every worker has opened */
  private acks = new Map<string, number>();
  /** T-1174: the items each worker has acknowledged (`acks` counts each worker once, since a
   * replacement reopens what its worker held) and the workers that have reported `ready` */
  private ackedBy: Set<string>[] = [];
  private isReady = new Set<number>();
  /** T-1195: workers that have opened each item inside a batch still opening (`packOpened`) */
  private packAcks = new Map<string, number>();
  private opened = new Set<string>();
  /** T-325: cities fetched to open bare (the aux worker only), and those open so, with their figures */
  private fetchingBare = new Set<CityId>();
  /** T-351: cities fetched before the reach was known, decided (bare or every worker) once downloaded */
  private fetchingUndecided = new Set<CityId>();
  private bareOpen = new Map<CityId, BareFigures>();
  /** T-301: the figures of zones open bare and closed since, still shown (no station reaches them) */
  private bareKept = new Map<CityId, BareFigures>();
  /** T-301: when each open item was last needed, and the sweep's next timed run */
  private lastNeeded = new Map<string, number>();
  private closeTimer = 0;
  /** bytes of each item, the weight of its opening in the percentage */
  private bytes = new Map<string, number>();
  private reach = new WeakMap<Service, Set<CityId>>();
  private nearZones = new WeakMap<Service, Near>();
  /** footprints being fetched for the reach test */
  private fetchingFootprints = new Set<CityId>();
  private onOpen: { item: string; f: (ok: boolean) => void }[] = [];
  /** the load the percentage describes: its items, the solve that completes it, its best percentage;
   * `since`, `wait`: T-884, when the wait the bar measures began (a game load's: its pick or load) and
   * which game wait it is (game/loading.ts `gameWait`); null for a load without a hold;
   * T-1195: `shares`, the bar's share for each stage (game/loadBar.ts), set when the load starts and
   * again when a game's network arrives (`modelled`), unless a remembered load set them; `netAt` and
   * `dataAt`, when the game stage and the data stage ended; T-1479: `modelSolve`, the solve model's
   * time for the network (before this browser's pace), `solveRounds` the solve's first and later rounds
   * as last seen, `left` the time left shown and when; T-2574: `viaCache`, the cache read expected to end the load
   * (`expectCache`), `cacheNet` whether its shares were set with the network in, `cacheReadyAt` when its packs and
   * network were in */
  private job: { items: string[]; cities: CityId[]; startup: boolean; version?: number; best: number; anchor?: { shown: number; frac: number }; since: number; wait: number | null; shares?: StageMs; modelled?: boolean; netAt?: number; dataAt?: number; pace?: Pace; modelData?: number; modelSolve?: number; solveRounds?: { first: number; rest: number; measured: boolean }; firstLand?: number[]; left?: { ms: number; at: number }; viaCache?: object; cacheNet?: boolean; cacheReadyAt?: number } | null = null;
  private progressTimer = 0;
  /** T-243: while a load runs the bar also moves with time, so progress is recomputed this often */
  private progressTick = 0;
  private endJob() {
    this.job = null;
    clearInterval(this.progressTick);
    this.progressTick = 0;
  }
  /** a loaded game's clock hold: released when this version has its first estimate */
  private hold: { version?: number } | null = null;
  /** T-2560: the network of a cached result whose hold went at the cache, until its solve's final or a newer
   * network: that solve keeps the hold's barrier, and the levels of detail wait for it */
  private cacheSolve: Service | null = null;

  // --- T-328: the last final result kept beside the save (workers/demandCache.ts) ---
  /** T-2794: not on a small device, which has no memory for the copies a write makes or a worker to write them;
   * T-2867: nor with intercity trips off, the choice of a player short of memory */
  private cacheOn = cacheAvailable() && !TEST && !mobileDevice() && !marketOff();
  /** off: no cache is written (a check turns it off to leave a network uncached) */
  cacheWrites = true;
  /** SHA-256 of each pack opened in every worker (and "market"), as it downloads */
  private packHash = new Map<string, { p: Promise<string>; set: (h: string) => void }>();
  /** the zones of the cached result shown, and its network's hash */
  private cachedCities = new Set<CityId>();
  private cachedHash: string | null = null;
  /** its bubbles, if it kept them */
  private cachedModes: StoredDemand["modes"] | null = null;
  /** T-4202: it is another demand code's result standing in, so the fresh final is written over it */
  private cachedStandIn = false;
  private cachedLocal = false;
  private cacheSeq = 0;
  /** what the last read and write did (window.twDemand.client.cacheLog with ?debug=1) */
  cacheLog: { at: number; ahead?: boolean; hash?: string; readMs?: number; miss?: string; standIn?: string; shownAt?: number; skip?: string; wrote?: { at: number; ms: number; bytes: number; ok: boolean; ahead?: boolean } } | null = null;
  /** every read's `cacheLog`, oldest first */
  cacheLogs: NonNullable<DemandClient["cacheLog"]>[] = [];

  constructor(_city: string) {
    effect(() => { activeCity.value; worldFrameEpoch.value; this.syncFocus(); });
    // T-2045: a zone newly in view during a game load is solved first, in the focus pool
    effect(() => { activeCity.value; viewedZone.value; zonesLoadForView.value; this.focusSoon(); });
    const n = (this.computeN = this.poolN = poolSize());
    solveWaves = Math.ceil(PERIODS.length / n);
    // i18n-ignore: a stored name or a developer diagnostic
    if (new URLSearchParams(location.search).has("debug")) console.info(`demand workers: ${n} (threads ${navigator.hardwareConcurrency}, memory ${(navigator as { deviceMemory?: number }).deviceMemory ?? "not reported"})`);
    if (this.holding) this.holding = new Holding(n);
    this.dealPeriods(n);
    // (a pool of one stays one: `growTo` grows to `poolN` at most)
    this.auxShared = n === 1 && phoneLoad() && !this.holding;
    for (let i = 0; i < n + (this.auxShared ? 0 : 1); i++) this.workers.push(this.spawn(i, i === 0 && n > 1));
    if (this.viewHeld) setTimeout(() => this.releaseView(), 10_000);
    // the viewed city loads when it changes (a pan, the city buttons)
    effect(() => { activeCity.value; viewedZone.value; zonesLoadForView.value; this.ensure(); });
    // and the zones beside it with bubbles on screen (T-276), once nothing waits on them
    effect(() => { nearbyZones.value; this.ensure(); });
    // T-801: a selected zone's market opens for its partner bubbles
    // T-2453: and the zone the city view shows goes first after the zone in view
    effect(() => { selection.value; this.selectedFirst(); this.ensure(); this.lodSoon(); queueMicrotask(() => this.lengthsAhead()); });
    // T-1223: the zone in view goes fine once it has been in view VIEW_SETTLE_MS
    effect(() => {
      const id = viewedZone.value && zonesLoadForView.value ? activeCity.value : null;
      if (id === this.viewAt.id) return;
      this.viewAt = { id, at: Date.now() };
      this.lodShown();
      this.lodSoon(VIEW_SETTLE_MS + 50);
    });
    // T-2180: the level changes held while edits settled go at once after
    effect(() => { if (!editsSettling.value) this.lodSoon(); });
    // T-4117: the player's setting changed: every open zone opens again at its new level
    let coarse = display.peek().coarseDemand;
    effect(() => { if (display.value.coarseDemand !== coarse) { coarse = display.peek().coarseDemand; this.lodSwitched(); } });
    // T-1047: zones built track crosses open bare for their figures
    effect(() => { trackZones.value; this.ensure(); });
    // T-351: a building tool picked in a game with no station readies the viewed zone for its solve
    effect(() => { tool.value; this.prefetchViewed(); });
    // T-1373: a superholiday's rush days near or gone redo the shown views with them or without
    effect(() => { rushRun.value; this.refreshDays(); });
    // T-1904: the day shown and the zones with bubbles, for the sums a solve sends unasked
    effect(() => {
      const v = this.viewMsg();
      for (const w of this.workers) w.postMessage(v);
      this.fpool?.postAll(v);
      this.apool?.postAll(v);
    });
    // T-1533: a game load's first figures for the zone in view show the map behind the card
    effect(() => {
      const seq = gameLoadSeq.value;
      if (seq !== this.figSeq) { this.figSeq = seq; this.figDays = demandByDay.peek(); }
      // (zoomed out, or over dead space, no zone is in view: any figures of the load)
      const inView = !!viewedZone.value && zonesLoadForView.value;
      const days = demandByDay.value;
      if (days !== this.figDays && (inView ? cityDemandViews.value.has(activeCity.value) : !!days[0]?.view)) viewFiguresIn();
    });
  }

  private figSeq = -1;
  private figDays: unknown = null;
  /** T-1904: the `view` message for the day and zones on screen now (read in an effect) */
  private viewMsg(): ToDemand {
    // (none with the bubbles off: Tokyo's take a worker about 0.3 s a round)
    return { kind: "view", day: shownDay.value, zones: display.value.commuters ? [activeCity.value, ...nearbyZones.value] : [] };
  }

  /** Worker `i`, its WASM started at once; packs follow as they arrive (T-149). Messages from a
   * worker replaced (T-1174) or retired (T-1205) since are dropped; the aux worker's index moves
   * when the pool shrinks. */
  private spawn(i: number, warmEveryday: boolean): Worker {
    const n = this.computeN;
    const w = new Worker(new URL("./demand.worker.ts", import.meta.url), { type: "module" });
    const at = () => this.workers.indexOf(w);
    w.onmessage = (e: MessageEvent<FromDemand>) => { const j = at(); if (j >= 0) this.receive(j, e.data); };
    // an uncaught WASM trap is a panic as `dead` reports one
    // (T-2478: a worker whose script did not download has an error with no message; T-2704: such a
    // worker, which never said anything, is tried again)
    let heard = false;
    w.addEventListener("message", () => { heard = true; }, { once: true });
    w.onerror = (e) => {
      const j = at();
      if (j < 0) return;
      if (!e.message && !heard) return this.scriptFailed(j);
      // i18n-ignore
      if (/RuntimeError|unreachable|recursive use of an object/.test(e.message ?? "") || isMemoryFailure(e.message ?? "")) this.workerDied(j, e.message);
      else this.fail(`worker ${j}: ${e.message || "its code could not be downloaded"}`); // i18n-ignore
    };
    w.onmessageerror = () => { const j = at(); if (j >= 0) this.fail(`worker ${j}: a message could not be read`); }; // i18n-ignore
    // T-325: the aux worker is told every solving worker's periods, for the zones it holds bare
    const msg: ToDemand = { kind: "init", packs: [], periods: this.periodsOf[i], ...(i < n ? this.marketsMsg(i) : {}), aux: i === n, warmEveryday, ...(i === n ? { groups: this.auxGroups() } : {}), ...(this.auxShared && i === 0 ? { bareToo: true, groups: this.auxGroups() } : {}), heapStep: heapStep(), ...SUM_FLAGS };
    w.postMessage(msg);
    // T-3781, dev server only: `?debugStall=i` has worker i's first solve wait forever (a replacement does not)
    if (import.meta.env.DEV && Number(new URLSearchParams(location.search).get("debugStall") ?? -1) === i && !this.stallSent) {
      this.stallSent = true;
      w.postMessage({ kind: "debugPanic", when: "stall" } satisfies ToDemand);
    }
    return w;
  }
  private stallSent = false;

  /**
   * T-351: in a game with no station (a bare start), a building tool picked opens the viewed zone,
   * open bare, in every worker too, so the first station's solve need not wait for it (about 2.4 s
   * for New York in five workers). No solve covers it for that (`solveNeed`); it joins the solves
   * once a station reaches it, as any zone. Not while a game load's card is up.
   */
  private prefetchViewed() {
    const t = tool.peek();
    if (t !== "track" && t !== "station" && t !== "blueprint") return;
    const svc = this.latest(), id = activeCity.peek();
    if (this.hold || !svc || svc.stations.length || !viewedZone.peek() || this.failed) return;
    if (!this.fetchingBare.has(id) || this.fetching.has(id) || this.fetchingUndecided.has(id)) return;
    this.prefetched.add(id);
    this.fetchCity(id, "full");
  }
  /** T-351: zones opened in every worker ahead of a station (`prefetchViewed`, `stationsAhead`) */
  private prefetched = new Set<CityId>();

  /**
   * T-889: a save being loaded has these built stations (x, y in `frame`). Working its network out
   * takes the clock worker several seconds on a big game (Japan's about 8), so the zones the stations
   * reach and the markets `wantsMarket` will want for them start downloading and opening now, as
   * `ensure` opens them once the network arrives. They count as prefetched, so no solve covers one
   * the network turns out not to reach; a market it does not want is closed (`settleAhead`).
   */
  stationsAhead(frame: FrameSpec, xy: Float64Array, fine: string[] | null = null, built: string[][] | null = null) {
    // T-1223: a game loading: its fine set is made afresh, from the save's list if it has one
    this.lodReset(fine, built);
    queueMicrotask(() => this.releaseView()); // after the camera's move to the save's view, which follows this call
    if (this.failed || xy.length < 2) return;
    this.endFocus(); // (a game picked while another loaded)
    this.endAside();
    this.dropSide();
    this.startFocus();
    const stations: Service["stations"] = [];
    for (let k = 0; k + 1 < xy.length; k += 2) stations.push({ id: "", x: xy[k], y: xy[k + 1] });
    const svc: Service = { frame, fare: { base: 0, perKm: 0 }, stations, lines: [] };
    this.ahead = { svc, markets: new Set() };
    void loadFootprints(this.nearOf(svc).near).then(() => {
      const a = this.ahead;
      if (a?.svc !== svc || this.failed) return;
      const reached = this.reachedSet(svc);
      if (!reached?.size) return;
      if (this.lod === "auto" && !this.lodZones) this.initFine([...this.withoutBroken(new Set(reached))]);
      // T-3373: a big save's pool is cut before its packs open, by their estimate
      const need = memoryNeed([...reached]);
      this.capAhead(need.wasmGB * 2 ** 30 + MEM_WORKER, need.auxGB * 2 ** 30);
      // T-1533: the zone in view's pack first (T-2453: then the selected zone's), then the markets,
      // then the other zones
      const firsts = this.firstZones();
      const rank = (id: CityId) => { const k = firsts.indexOf(id); return k < 0 ? firsts.length : k; };
      const zones = [...this.withoutBroken(new Set(reached))].sort((x, y) => rank(x) - rank(y));
      const fetchZone = (id: CityId) => {
        if (this.fetching.has(id) || this.fetchingUndecided.has(id)) return;
        this.prefetched.add(id);
        this.fetchCity(id, "full");
      };
      for (const id of zones) if (rank(id) < firsts.length) fetchZone(id);
      for (const id of marketOff() ? [] : new Set([...reached, ...this.nearOf(svc).markets])) {
        const item = marketItem(id);
        if (this.fetching.has(item) || this.broken.has(item)) continue;
        a.markets.add(id);
        this.fetchMarket(id);
      }
      for (const id of zones) fetchZone(id);
      this.focusSoon(); // (T-2533: the focus pool's pack too)
      this.progress();
    });
  }
  /** `arrived`: the loaded game's network is in */
  private ahead: { svc: Service; markets: Set<CityId>; arrived?: boolean } | null = null;

  /** T-889: the loaded game's network has arrived with its reach (`reached`): markets opened ahead
   * that it does not want close, before a solve of it would count them. */
  private settleAhead(reached: Set<CityId>) {
    const a = this.ahead;
    if (!a?.arrived) return;
    this.ahead = null;
    const want = reached.size ? this.marketZones(this.latest()) : new Set<CityId>();
    const extra = [...a.markets].filter((id) => !want.has(id));
    if (!extra.length) return;
    console.warn("demand: markets opened ahead that the network does not want", extra);
    const open = extra.filter((id) => this.opened.has(marketItem(id)));
    if (open.length) this.closeFull([], open);
  }

  /** T-276: nearby zones open for their bubbles only after the startup load and outside a loaded
   * game's clock hold, so they never hold the clock; they show no loading bar and a solve never
   * waits for them. T-325: they open bare unless a station reaches them. */
  private nearbyOpen(): boolean {
    return startup.peek() === "ready" && !this.hold;
  }

  // --- Loading only what is needed (T-149) -------------------------------------------------------

  /** T-318: the viewed zone alone; empty over dead space, where nothing loads for the view, and
   * (T-871) zoomed out past `ZONE_LOAD_ZOOM`. */
  private viewedSet(): Set<CityId> {
    if (this.viewHeld) return new Set();
    return this.withoutBroken(new Set(viewedZone.peek() && zonesLoadForView.peek() ? [activeCity.peek()] : []));
  }

  /** T-899: a saved game's pick has not yet moved the camera to the save's own view, so the zone in view is
   * only the page's default (New York): nothing loads for the view until the save's stations or network arrive
   * (or after 10 s, if neither does). */
  private viewHeld = SAVED_GAME;
  private releaseView() {
    if (!this.viewHeld) return;
    this.viewHeld = false;
    this.ensure();
    this.focusSoon();
  }

  /** T-2453: the zone the city view shows (a city selection), when it is not the zone in view */
  private selectedZone(): CityId | null {
    const sel = selection.peek();
    return !perfOff("selected") && sel?.kind === "city" && sel.city !== activeCity.peek() ? (sel.city as CityId) : null;
  }
  /** T-1533, T-2453: the zones that go first through every queue, in order: the zone in view, then
   * the zone the city view shows */
  private firstZones(): CityId[] {
    const sel = this.selectedZone();
    return sel ? [activeCity.peek(), sel] : [activeCity.peek()];
  }

  /** T-318: a network is in hand and neither the view nor any station needs a pack: nothing to
   * load or solve. The clock's hold goes, and no card or status says "Loading data". A waiting
   * network with stations is still solved, with no pack (T-845): its lines carry nobody, and
   * without that solve they never get figures. */
  private idle(): boolean {
    const svc = this.latest();
    return !!svc && this.needed(svc)?.size === 0 && !this.waiting?.svc.stations.length;
  }

  private settleIdle() {
    demandUpdating.value = false; // a waiting network stays: it is solved when the view or a station needs a pack
    finalWorld.value = networkWorld.peek(); // (T-2658: nothing in it to solve)
    this.uncovered = [];
    demandBehindFrom.value = null;
    this.endJob();
    loadProgress.value = null;
    const held = !!this.hold || startup.peek() === "loading";
    this.hold = null;
    if (held) loadComplete();
    if (held && trackZones.peek().built.size) this.ensure(); // T-1047
  }

  /** The cities a solve of `svc` needs open: the viewed one, and every one a built station reaches.
   * Null while that is not known yet (the footprints are still loading). */
  private needed(svc: Service | null): Set<CityId> | null {
    const out = this.viewedSet();
    if (!svc) return this.withoutBroken(out);
    const reached = this.reachedSet(svc);
    if (!reached) return null;
    for (const id of reached) out.add(id);
    return this.withoutBroken(out);
  }

  /** The cities `svc`'s stations reach; null while that is not known (no network yet, or the
   * footprints of zones whose boxes the stations come near still loading; T-404). A zone whose
   * footprint failed counts as reached. */
  private reachedSet(svc: Service | null): Set<CityId> | null {
    if (!svc) return null;
    if (svc.stations.length === 0 && !svc.alone?.length) return new Set();
    let reached = this.reach.get(svc);
    if (reached) return reached;
    const { ll, near, key } = this.nearOf(svc);
    // (a timetable edit is a new service with the same stations)
    if (this.reachLast?.key === key) {
      this.reach.set(svc, this.reachLast.reached);
      return this.reachLast.reached;
    }
    const missing = near.filter((id) => footprintOf(id) === undefined);
    if (missing.length) {
      const fresh = missing.filter((id) => !this.fetchingFootprints.has(id));
      for (const id of fresh) this.fetchingFootprints.add(id);
      if (fresh.length) void loadFootprints(fresh).then(() => {
        for (const id of fresh) this.fetchingFootprints.delete(id);
        this.ensure();
      });
      return null;
    }
    reached = new Set();
    const fps: Footprint[] = [];
    for (const id of near) {
      const f = footprintOf(id);
      if (f) fps.push(f);
      else reached.add(id);
    }
    for (const id of citiesReached(fps, ll)) reached.add(id);
    this.reach.set(svc, reached);
    this.reachLast = fps.length === near.length ? { key, reached } : null;
    return reached;
  }
  /** the last reach worked out, by its stations' positions (`nearOf`'s key); none kept while a
   * footprint has failed, since it may load later and change the reach */
  private reachLast: { key: string; reached: Set<CityId> } | null = null;

  private nearOf(svc: Service): Near {
    let o = this.nearZones.get(svc);
    if (!o) {
      const at = [...svc.stations.map((s): [number, number] => [s.x, s.y]), ...(svc.alone ?? [])];
      const key = `${JSON.stringify(svc.frame)}|${at.join(";")}`;
      if (this.nearLast?.key === key) o = this.nearLast;
      else {
        const f = makeFrame(svc.frame), ll = new Float64Array(2 * at.length);
        const markets = new Set<CityId>();
        at.forEach(([x, y], k) => {
          const [lon, lat] = f.toLonLat(x, y);
          ll[2 * k] = lon; ll[2 * k + 1] = lat;
          for (const id of marketZonesNear(lon, lat)) markets.add(id);
        });
        o = this.nearLast = { key, ll, near: zonesNearStations(ll, REACH_M), markets: [...markets] };
      }
      this.nearZones.set(svc, o);
    }
    return o;
  }
  private nearLast: Near | null = null;

  /** T-325: the cities a solve of `svc` must cover: every one its stations reach, and every one
   * opened in all workers (the viewed city at startup, before the reach was known). Null while
   * the reach is not known. */
  private solveNeed(svc: Service | null): Set<CityId> | null {
    const reached = this.reachedSet(svc);
    if (!reached) return null;
    const out = new Set(reached);
    // (T-351: not a zone opened ahead of a station that does not reach it yet)
    for (const id of this.fetching) if (!isMarket(id) && !this.prefetched.has(id as CityId)) out.add(id as CityId);
    return this.withoutBroken(out);
  }

  /** Open in every worker, or bare in the aux worker (T-325). */
  private isOpen(item: string): boolean {
    return this.opened.has(item) || this.bareOpen.has(item as CityId);
  }

  /** T-325: `city` is answered by the aux worker's bare copy: open bare, and not covered by the
   * solve shown (a city reached since stays bare until a shown solve covers it). */
  private isBare(city: CityId): boolean {
    if (!this.bareOpen.has(city)) return false;
    const v = demandView.peek()?.version;
    return !this.log.find((s) => s.version === v)?.cities.includes(city);
  }

  /** The newest service: waiting for packs, or the one being solved. */
  private latest(): Service | null {
    return this.waiting?.svc ?? this.cur?.svc ?? null;
  }

  /** Every city in `need` and the market packs wanted are open in every worker. */
  private allOpen(need: Set<CityId>): boolean {
    return this.ready === this.workers.length && this.marketWanted.every((id) => this.opened.has(marketItem(id))) && [...need].every((id) => this.opened.has(id));
  }

  /** T-396, T-377, T-801: a zone's market pack is opened once the zone is reached, a station is
   * within its riders' reach (`marketZonesNear`), or it is selected (a zone, or a bubble selection in
   * it); never for viewing alone, since each is tens of MB in every solving worker. It closes as a
   * city pack does (`sweep`). */
  private marketWanted: CityId[] = [];
  private marketZones(svc: Service | null): Set<CityId> {
    // T-2802: a small device opens no market: the world's rows and a zone's pack took its one solving worker from 33 to 184 MB on a first line
    if (marketOff()) return new Set();
    const sel = selection.peek();
    // T-979: a trip with an end in another zone is read from the market too
    const leaves = sel?.kind === "trip" && !!sel.to && (!!sel.far || [sel.from, sel.to].some(([x, y]) => zoneAt(...packToLngLat(x, y)) !== activeCity.peek()));
    const picked = sel?.kind === "city" ? sel.city : sel?.kind === "commuters" || leaves ? activeCity.peek() : null;
    return new Set<CityId>([...(this.knownReach() ?? []), ...(svc ? this.nearOf(svc).markets : []), ...(picked ? [picked] : [])]);
  }
  private wantsMarket(svc: Service | null): CityId[] {
    const out = this.marketZones(svc);
    this.marketWanted = CITIES.map((c) => c.id).filter((id) => out.has(id) && !this.broken.has(marketItem(id)));
    return this.marketWanted;
  }

  /** Fetch whatever the newest service and the view need, the viewed city first; then solve if all is open. */
  private ensure() {
    if (this.failed) return;
    const known = this.needed(this.latest());
    const need = known ?? this.viewedSet();
    // T-318; T-889: not while a loaded game's network is on its way (the camera moving to its view
    // may pass over dead space, which hid the card for a moment)
    demandIdle.value = known?.size === 0 && !gameLoadArmed();
    // T-325: a city no station reaches opens bare; while the reach is not known (startup), the
    // viewed city's pack is fetched and decided once downloaded (T-351)
    const reached = this.knownReach();
    // T-1223: a game's first fine set before any of its packs is sent (not a loading save's before its
    // network arrives: the reach known now is the last game's)
    if (this.lod === "auto" && !this.lodZones && reached && (!this.ahead || this.ahead.arrived)) this.initFine([...this.withoutBroken(new Set(reached))]);
    // (the loaded network's reach can pass what its stations ahead gave: those zones are the load's
    // too, coarse unless the first fine set has them)
    else if (this.lod === "auto" && this.lodZones && reached && this.hold && (!this.ahead || this.ahead.arrived)) for (const id of reached) this.lodZones.add(id);
    // T-1533: the zone in view's pack first, ahead of the markets and the other zones
    // (T-2453: then the selected zone's)
    for (const id of this.firstZones()) if (need.has(id)) this.want(id, reached === null ? null : reached.has(id));
    // over dead space with no station in reach there is nothing to open, the market included
    if (need.size) for (const id of this.wantsMarket(this.latest())) if (!this.fetching.has(marketItem(id))) this.fetchMarket(id);
    if (known) this.settleAhead(known);
    // T-729: the save's list of zones reached
    if (reached?.size) addOpened(reached);
    for (const id of need) this.want(id, reached === null ? null : reached.has(id));
    if (this.nearbyOpen()) for (const id of nearbyZones.peek()) this.want(id, !!reached?.has(id));
    this.openCrossed(reached);
    this.flush();
    this.progress();
    this.sweep();
  }

  // --- Level of detail (T-1223, notes/T-1206.md) -------------------------------------------------

  private get lod() { return lodMode(display.peek().coarseDemand); }
  /** the zones wanted fine; `lodZones` the zones the last check decided for (a zone reached since
   * opens fine: a station was just built there) */
  private fine = new Set<CityId>();
  private lodZones: Set<CityId> | null = null;
  /** the save's `demandFine` (null: it has none), until the first fine set is made */
  private savedFine: string[] | null = null;
  /** the level each zone's pack was last sent to the solving workers at */
  private levelPosted = new Map<CityId, number>();
  /** zones being opened again at another level: open as far as loading goes */
  private releveling = new Set<CityId>();
  /** wall ms each zone was last in priorities 1-3, was made fine, was edited */
  private touched = new Map<CityId, number>();
  private madeFine = new Map<CityId, number>();
  private edited = new Map<CityId, number>();
  /** T-2005: wall ms each zone last had a station placed or a stop added (never pruned); a loaded
   * save's last two builds (`demandBuilt`) start at 2 and 1, before any build of this session */
  private built = new Map<CityId, number>();
  private viewAt: { id: CityId | null; at: number } = { id: null, at: 0 };
  private lodTimer = 0;
  private lodTick = 0;

  /** The level zone `id` is to be opened at now. */
  private levelFor(id: CityId): number {
    if (this.lod !== "auto") return this.lod === "fine" ? LOD_FINE : LOD_COARSE;
    return !this.lodZones?.has(id) || this.fine.has(id) ? LOD_FINE : LOD_COARSE;
  }
  private coarseOf(cities: CityId[]): CityId[] {
    return cities.filter((id) => this.levelPosted.get(id) === LOD_COARSE);
  }

  /** Priorities 1-3 (notes/T-1206.md section 3), highest first: the zone in view once settled, the
   * selected zones, zones edited within EDIT_KEEP_MS. */
  private heldZones(now: number): CityId[] {
    const out: CityId[] = [];
    const add = (id: CityId | null | undefined) => { if (id && !out.includes(id)) out.push(id); };
    const v = this.viewedSet().has(activeCity.peek()) ? activeCity.peek() : null;
    // (T-4103: on a phone the view keeps a fine zone fine but never refines a coarse one: about 140 MB for a minute
    // after a pan, for a few MB of detail)
    if (v && this.viewAt.id === v && now - this.viewAt.at >= VIEW_SETTLE_MS && (!phoneLoad() || this.fine.has(v))) add(v);
    const sel = selection.peek();
    const at = (x: number, y: number) => zoneAt(...packToLngLat(x, y));
    if (sel?.kind === "city") add(sel.city as CityId);
    else if (sel?.kind === "commuters") add(activeCity.peek());
    else if (sel?.kind === "trip") { add(at(...sel.from)); if (sel.to) add(at(...sel.to)); }
    else if (sel?.kind === "station") {
      const s = world.peek()?.save.stations.find((x) => x.id === sel.station);
      if (s) add(zoneAt(s.lng, s.lat));
    }
    for (const [id, t] of [...this.edited].sort((a, b) => b[1] - a[1])) if (now - t < EDIT_KEEP_MS) add(id);
    return out;
  }

  /** Check soon (a view settling, a click, an edit). */
  private lodSoon(ms = 0) {
    if (this.lod === "fine") return;
    clearTimeout(this.lodTimer);
    this.lodTimer = setTimeout(() => this.lodCheck(), ms) as unknown as number;
    if (!this.lodTick && this.lod === "auto") this.lodTick = setInterval(() => this.lodCheck(), 30_000) as unknown as number;
  }

  /** The zones a solve covers, once their reach is known. */
  private lodReach(): CityId[] | null {
    const r = this.knownReach();
    return r ? [...this.withoutBroken(new Set(r))] : null;
  }

  /** A game is loading: its fine set is made afresh once its reach is known (`saved`: its save's list;
   * `built`: its save's last two builds, latest first). */
  private lodReset(saved: string[] | null, built: string[][] | null = null) {
    this.lodZones = null;
    this.lastReg.clear();
    this.lastAir.clear();
    this.savedFine = saved;
    this.fine = new Set();
    this.touched.clear();
    this.madeFine.clear();
    this.edited.clear();
    this.built.clear();
    (built ?? []).slice(0, 2).forEach((ids, i) => { for (const id of ids) if (isCity(id) && !this.built.has(id as CityId)) this.built.set(id as CityId, 2 - i); });
  }

  /** A game's first fine set (`firstFine`), from the save's list or the budget. */
  private initFine(zones: CityId[]) {
    const c = viewCentre.peek() ?? cityById(activeCity.peek())?.center ?? [0, 0], k = Math.cos((c[1] * Math.PI) / 180);
    const d = (id: CityId) => { const z = cityById(id); return z ? ((z.center[0] - c[0]) * k) ** 2 + (z.center[1] - c[1]) ** 2 : Infinity; };
    const order = [activeCity.peek(), ...nearbyZones.peek(), ...zones.slice().sort((a, b) => d(a) - d(b))];
    this.fine = firstFine(zones, order, this.savedFine, lodBudgetGB());
    this.lodZones = new Set(zones);
    const now = Date.now();
    for (const id of this.fine) { this.touched.set(id, now); this.madeFine.set(id, now); }
    this.savedFine = null;
    if (DEBUG_LOD) console.info(`T1223 first fine set: ${this.fine.size} of ${zones.length} zones fine (${[...this.fine].slice(0, 8).join(",")})`);
  }

  /** Work the fine set out again and open the zones whose level changed at their new one. */
  private lodCheck() {
    clearTimeout(this.lodTimer);
    this.lodTimer = 0;
    // (a game loading makes its first fine set from its own reach, in `ensure` or `stationsAhead`)
    if (this.lod === "fine" || this.failed || this.hold || this.cacheSolve || gameLoadArmed() || (this.ahead && !this.ahead.arrived)) return;
    if (this.lod === "auto" && !this.lodZones) return;
    const now = Date.now();
    // T-2180: no level changes while edits settle (the edit note's figures would mix them in), and
    // that time counts as looked at, so a fine zone's ten minutes left alone start after
    if (editsSettling.peek()) {
      for (const id of this.fine) this.touched.set(id, now);
      return;
    }
    if (this.lod === "coarse") return void this.lodApply();
    const zones = this.lodReach();
    if (!zones) return;
    const held = this.heldZones(now);
    for (const id of held) this.touched.set(id, now);
    // a zone reached since the last check counts as just made fine (it opened fine)
    const fresh = zones.filter((id) => !this.lodZones!.has(id));
    for (const id of fresh) { this.fine.add(id); this.touched.set(id, now); this.madeFine.set(id, now); }
    const keep = builtKeep(this.built, new Set(zones));
    const next = planFine({ zones, held, near: nearbyZones.peek(), fine: this.fine, touched: this.touched, madeFine: this.madeFine, now, budget: lodBudgetGB(), keep });
    for (const id of next) if (!this.fine.has(id)) this.madeFine.set(id, now);
    for (const id of [...this.edited.keys()]) if (now - this.edited.get(id)! >= EDIT_KEEP_MS) this.edited.delete(id);
    for (const id of [...this.touched.keys()]) if (!next.has(id) && now - this.touched.get(id)! >= LEFT_ALONE_MS) this.touched.delete(id);
    this.fine = next;
    this.lodZones = new Set(zones);
    reportDemandFine(zones.filter((id) => next.has(id)), keep);
    this.lodApply();
  }

  /** T-4117: the setting changed. Back to the rules, the fine set is made afresh as a game's first (the save's list
   * while this game has not made one), then the next check opens each zone at its level; coarse, every zone goes
   * coarse at that check. A loading game's card holds the change until it goes. */
  private lodSwitched() {
    if (this.lod === "auto") {
      this.lodZones = null;
      const zones = this.lodReach();
      if (zones) this.initFine(zones);
    }
    this.lodSoon();
  }

  /** Open again, in every worker, each zone open at a level it should no longer have: zones that
   * change together in one check do so together, so one solve covers them (`flush`). */
  private lodApply() {
    const change = [...this.levelPosted].filter(([id, level]) => this.opened.has(id) && !this.releveling.has(id) && this.fetching.has(id) && level !== this.levelFor(id)).map(([id]) => id);
    if (DEBUG_LOD && change.length) console.info(`T1223 levels change: ${change.map((id) => `${id}:${this.levelFor(id)}`).join(",")}`);
    for (const id of change) {
      this.releveling.add(id);
      this.opened.delete(id);
      this.unack(id);
      this.packAcks.delete(id);
      // (T-4265: the aux worker keeps its copy, fine at every level, and the cells come from it: only the squares, from a
      // solving worker's, go)
      if (this.auxShared) this.dropCells(id);
      else this.zsq.delete(id);
      this.fetchCity(id, "full");
    }
    this.lodShown();
    if (change.length) this.refineAside();
  }

  /** The zones whose shown figures are coarse (`roughZones`) and the zone in view being worked out
   * in detail (`refiningZone`). `cities` and `coarse`: a solve's zones just shown and which of them
   * were coarse. */
  private lodShown(cities?: CityId[], coarse?: CityId[]) {
    if (cities) {
      const c = new Set(coarse ?? []), next = new Set(roughZones.peek());
      let changed = false;
      for (const id of cities) if (c.has(id) !== next.has(id)) { changed = true; if (c.has(id)) next.add(id); else next.delete(id); }
      if (changed) roughZones.value = next;
    }
    const v = activeCity.peek();
    const refining = roughZones.peek().has(v) && this.levelPosted.has(v) && this.levelFor(v) === LOD_FINE ? v : null;
    if (refiningZone.peek() !== refining) refiningZone.value = refining;
  }

  /** T-1223, `?debug=1`: the fine set and levels, for checks. */
  lodState() {
    const zones = this.lodReach() ?? [];
    const open = [...this.levelPosted].filter(([id]) => this.opened.has(id));
    return { mode: this.lod, fine: [...this.fine], fineOpen: open.filter(([, l]) => l === LOD_FINE).map(([id]) => id), coarseOpen: open.filter(([, l]) => l === LOD_COARSE).map(([id]) => id), releveling: [...this.releveling], refining: refiningZone.peek(), rough: [...roughZones.peek()], built: [...this.built].sort((a, b) => b[1] - a[1]).map(([id]) => id), zonesGB: zonesGB(zones, this.fine), budgetGB: lodBudgetGB() };
  }

  /** The zones a new network changed (stations added, removed or moved; lines whose stops, trains or
   * cars changed: the zones of the stops that changed, else of the stops in the zone in view). */
  private noteEdits(prev: Service, next: Service) {
    const f = makeFrame(next.frame), zone = (s: { x: number; y: number }) => zoneAt(...f.toLonLat(s.x, s.y));
    const was = new Map(prev.stations.map((s) => [s.id, s])), now = new Map(next.stations.map((s) => [s.id, s]));
    const zones = new Set<CityId | null>(), built = new Set<CityId | null>();
    for (const s of next.stations) { const o = was.get(s.id); if (!o || o.x !== s.x || o.y !== s.y) { zones.add(zone(s)); built.add(zone(s)); } }
    for (const s of prev.stations) if (!now.has(s.id)) zones.add(zone(s));
    const lines = new Map(prev.lines.map((l) => [l.id, l]));
    const view = activeCity.peek();
    for (const l of next.lines) {
      const o = lines.get(l.id);
      if (o && o.stops.join() === l.stops.join() && o.tph.join() === l.tph.join() && o.cars === l.cars) continue;
      const had = new Set(o?.stops ?? []), has = new Set(l.stops);
      const moved = [...l.stops.filter((s) => !had.has(s)), ...(o?.stops ?? []).filter((s) => !has.has(s))];
      const at = (id: string) => { const s = now.get(id) ?? was.get(id); return s ? zone(s) : null; };
      if (moved.length) for (const id of moved) zones.add(at(id));
      else if (l.stops.some((id) => at(id) === view)) zones.add(view);
      for (const id of l.stops) if (!had.has(id)) built.add(at(id));
    }
    const t = Date.now();
    for (const id of zones) if (id) this.edited.set(id, t);
    for (const id of built) if (id) this.built.set(id, t);
  }

  // --- Closing what is no longer needed (T-301) ---------------------------------------------------

  /** What must stay open: the zones viewed, beside the view, reached or in the load in hand, and the
   * markets `marketZones` wants. Null while the reach is not known. */
  private keepSet(): Set<string> | null {
    const reached = this.knownReach();
    if (!reached) return null;
    const keep = new Set<string>([...this.viewedSet(), ...nearbyZones.peek(), ...reached]);
    if (this.job) for (const id of this.job.items) keep.add(id);
    for (const id of this.marketZones(this.latest())) keep.add(marketItem(id));
    // T-4202: while a save loaded in a running game has its network on the way, the reach above is the last game's: the
    // packs opening ahead for the save (`stationsAhead`) stay, or each worker opens them twice
    const a = this.ahead;
    if (a && !a.arrived) {
      for (const id of this.prefetched) keep.add(id);
      for (const id of a.markets) keep.add(marketItem(id));
    }
    return keep;
  }

  /** Close the packs no longer needed (CLOSE_MS, KEEP_SPARE), and bare copies the solves have made
   * redundant. Not before startup has finished nor in a loaded game's hold; `?perfOff=close` never. */
  private sweep() {
    clearTimeout(this.closeTimer);
    this.closeTimer = 0;
    if (perfOff("close") || this.failed || startup.peek() === "failed") return;
    // (T-1366: tried again after, as nothing else may ask: a still view in a paused game, an earlier game's markets)
    if (this.ready < this.workers.length || this.held || startup.peek() !== "ready" || this.hold) return void (this.closeTimer = setTimeout(() => this.sweep(), MEM_RETRY_MS) as unknown as number);
    const keep = this.keepSet();
    if (!keep) return;
    const now = performance.now();
    for (const id of keep) this.lastNeeded.set(id, now);
    const last = (id: string) => this.lastNeeded.get(id) ?? 0;
    // a zone open bare and in every worker whose shown solve covers it: the bare copy answers nothing
    const redundant = [...this.bareOpen.keys()].filter((id) => this.opened.has(id) && !this.isBare(id));
    if (redundant.length) this.closeBare(redundant);
    const inFlight = (id: CityId) => this.fetchingUndecided.has(id) || (this.fetching.has(id) && !this.opened.has(id)) || (this.fetchingBare.has(id) && !this.bareOpen.has(id));
    const spare = CITIES.map((c) => c.id)
      .filter((id) => !keep.has(id) && (this.opened.has(id) || this.bareOpen.has(id)) && !inFlight(id))
      .sort((a, b) => last(b) - last(a));
    const full: CityId[] = [], bare: CityId[] = [];
    let next = Infinity;
    spare.forEach((id, i) => {
      if (i >= KEEP_SPARE || now - last(id) >= CLOSE_MS) (this.opened.has(id) ? full : bare).push(id);
      else next = Math.min(next, last(id) + CLOSE_MS);
    });
    const markets: CityId[] = [];
    for (const item of this.opened) {
      if (!isMarket(item) || keep.has(item) || (this.fetching.has(item) && !this.opened.has(item))) continue;
      if (now - last(item) >= CLOSE_MS) markets.push(marketZone(item));
      else next = Math.min(next, last(item) + CLOSE_MS);
    }
    if (bare.length) this.closeBare(bare);
    if (full.length || markets.length) this.closeFull(full, markets);
    if (next < Infinity) this.closeTimer = setTimeout(() => this.sweep(), Math.max(1000, next - now + 50)) as unknown as number;
  }

  /** The load log's last entry of `item` (open bare, or in every worker) closed now, or opened now
   * unless it opened before. */
  private logLoad(item: string, bare: boolean, at: "openedAt" | "closedAt") {
    const log = lastLoad(this.start.loads, (l) => l.item === item && !!l.bare === bare);
    if (log && (at === "closedAt" || log.openedAt === undefined)) log[at] = performance.now();
  }

  /** Close zones open bare in the aux worker; their figures stay shown. */
  private closeBare(ids: CityId[]) {
    this.workers[this.auxAt].postMessage({ kind: "close", ids, bare: true } satisfies ToDemand);
    for (const id of ids) {
      this.bareKept.set(id, this.bareOpen.get(id)!);
      this.bareOpen.delete(id);
      this.fetchingBare.delete(id);
      this.dropCells(id);
      this.logLoad(id, true, "closedAt");
    }
  }

  /**
   * Close packs open in every worker (and their bare copies). No station reaches them, so they carry
   * no rider on any line and the shown figures stand: the solves stop counting them as covered, and
   * the next solve leaves them out. A worker solving with one closes it when that solve ends.
   */
  private closeFull(ids: CityId[], markets: CityId[]) {
    for (const w of this.workers) w.postMessage({ kind: "close", ids, ...(markets.length ? { markets } : {}) } satisfies ToDemand);
    const items: string[] = [...ids, ...markets.map(marketItem)];
    for (const id of items) {
      this.opened.delete(id);
      this.unack(id);
      this.packAcks.delete(id);
      this.fetching.delete(id);
      this.prefetched.delete(id as CityId);
      this.cityVersion.delete(id as CityId);
      this.levelPosted.delete(id as CityId);
      this.releveling.delete(id as CityId);
      this.dropCells(id);
      this.logLoad(id, false, "closedAt");
    }
    for (const s of this.log) {
      s.cities = s.cities.filter((id) => !ids.includes(id));
      s.markets = s.markets.filter((id) => !markets.includes(id));
    }
    const alsoBare = ids.filter((id) => this.bareOpen.has(id));
    if (alsoBare.length) this.closeBare(alsoBare);
  }

  /** T-351: the reach as far as opening a pack goes. A new game's (`?new=1`) is known to be empty
   * before its network arrives: it has no stations. */
  private knownReach(): Set<CityId> | null {
    const svc = this.latest();
    if (!svc && NEW_GAME && !this.lastKey) return new Set();
    return this.reachedSet(svc);
  }

  /**
   * T-1047: a zone built track crosses (game/trackZones.ts) that no station reaches opens bare, for
   * the figures its bubble and city view show: no station is in reach of its people, so a solve would
   * give it the same figures (T-149) at a pack in every worker. One at a time, nearest the view first,
   * once nothing else is opening bare and outside a load, so the view's zones keep the aux worker
   * first. Nothing keeps it open: it closes as a spare zone does and its figures stay (`bareKept`).
   */
  private openCrossed(reached: Set<CityId> | null) {
    if (!reached || !this.nearbyOpen()) return;
    const busy = (id: CityId) => this.fetchingUndecided.has(id) || (this.fetchingBare.has(id) && !this.bareOpen.has(id));
    if ([...this.fetchingBare, ...this.fetchingUndecided].some(busy)) return;
    const c = viewCentre.peek() ?? [0, 0], k = Math.cos((c[1] * Math.PI) / 180);
    let best: CityId | null = null, bd = Infinity;
    for (const id of trackZones.peek().built) {
      if (reached.has(id) || this.broken.has(id) || this.isOpen(id) || this.bareKept.has(id) || this.fetching.has(id)) continue;
      const z = cityById(id);
      const d = z ? ((z.center[0] - c[0]) * k) ** 2 + (z.center[1] - c[1]) ** 2 : Infinity;
      if (best === null || d < bd) { best = id; bd = d; }
    }
    if (best !== null) this.fetchCity(best, "bare");
  }

  /** Fetch `id` to open in every worker (`full`), bare in the aux worker (false), or (null: the
   * reach is not known yet) either, decided once downloaded; unless already. */
  private want(id: CityId, full: boolean | null) {
    if (this.fetching.has(id) || this.fetchingUndecided.has(id) || this.broken.has(id)) return;
    if (full) this.fetchCity(id, "full");
    else if (full === null) this.fetchCity(id, this.fetchingBare.has(id) ? "full" : "undecided");
    else if (!this.fetchingBare.has(id)) this.fetchCity(id, "bare");
  }

  /** Download one pack file, tried again after a failure (T-484: a server restarting or a pack
   * being replaced fails a fetch for a moment). */
  private async download(file: string, high = false): Promise<Uint8Array> {
    for (let attempt = 0; ; attempt++) {
      try {
        return await this.downloadOnce(file, high);
      } catch (err) {
        if (attempt >= RETRY_MS.length) throw err;
        await new Promise((r) => setTimeout(r, RETRY_MS[attempt]));
      }
    }
  }

  /** A city pack's header and .bin from the same build of it: the header is read again after the
   * .bin and they are fetched again if it changed meanwhile (T-484: a pack rewritten mid-load). */
  private async downloadPack(id: CityId): Promise<[Uint8Array, Uint8Array]> {
    for (let attempt = 0; ; attempt++) {
      // T-1533: the zone in view's pack ahead of the other downloads waiting for a connection
      const high = id === activeCity.peek();
      const [header, bin] = await Promise.all([this.download(`${id}.json`, high), this.download(`${id}.bin`, high)]);
      const again = await this.download(`${id}.json`, high);
      if (sameBytes(header, again)) {
        // T-2486: a published pack's .bin is gzip members; the workers and the sim get the raw layout
        const z = binGzOf(header);
        // a packed .bin under a header without the index (from another upload) would never open: a failed load instead
        if (!z && isGzip(bin)) throw new Error(`${id}.bin is packed but ${id}.json has no bin_gz`);
        return [header, z ? await unpackBin(bin, z) : bin];
      }
      if (attempt >= RETRY_MS.length) throw new Error(`${id}.json kept changing while it loaded`);
      await new Promise((r) => setTimeout(r, RETRY_MS[attempt]));
    }
  }

  /** Download one pack file once, counting its bytes as they arrive. */
  private async downloadOnce(file: string, high = false): Promise<Uint8Array> {
    const token = {};
    this.downloadsOpen.set(token, performance.now());
    try {
      return await this.fetchPackFile(file, high);
    } finally {
      this.downloadsOpen.delete(token);
    }
  }
  /** T-2756: when each pack download in flight started (before its headers came), for the status tag's slow-connection line */
  private downloadsOpen = new Map<object, number>();
  /** T-3313: the last worker panic or replacement, for a feedback report */
  lastTrouble = "";
  /** T-3781: each worker's last message (when, its kind), and the barrier solve period it started and has not
   * sent the result of; the status a failure replaced; when the stuck load's workers were restarted */
  private heard: ({ at: number; kind: string } | undefined)[] = [];
  private inUnit: ({ version: number; round: number; period: number; at: number } | null | undefined)[] = [];
  /** T-4343: the longest period any worker has solved on this page, start to result, ms, and the solve the last one
   * to end was of */
  private unitMsMax = 0;
  private unitTimedIn = -1;
  private statusBefore = "";
  private stuckRestarted = 0;
  /** T-4343: the quiet the loading card's watchdog allows while a solving worker is inside a period: a worker says
   * nothing from a period's start to its result, and on a big save solved on one worker a period runs for minutes
   * (past 8 on a 0.4.6 player's laptop), so a quiet that long is the work. UNIT_QUIET_SLACK times the longest period
   * this page has timed once one of this solve's has ended (an earlier solve's may have taken a millisecond), else
   * UNIT_QUIET_FIRST_MS; 0 when no worker is inside one of this solve's. */
  unitQuietMs(): number {
    const v = this.cur?.version;
    if (!this.inUnit.some((u, i) => u && i < this.computeN && u.version === v)) return 0;
    return this.unitTimedIn === v ? UNIT_QUIET_SLACK * this.unitMsMax : UNIT_QUIET_FIRST_MS;
  }

  /** T-3313: where the pool stands, one line for a feedback report: the newest solve's age and how far it got,
   * the last step, the oldest download in flight and what is still being fetched (seconds) */
  diagnosis(): string {
    const now = performance.now(), s = (t: number | undefined) => (t === undefined ? "-" : Math.round((now - t) / 1000) + "s");
    const c = this.cur, v = demandView.peek(), oldest = this.oldestDownload();
    const open = [...this.downloads].filter(([, d]) => d.end === undefined).map(([f, d]) => `${f} ${Math.round(d.got / 1024)}k ${s(d.at)}`);
    // T-3781: while round 0 is not published, the periods of it that came; each worker's last message and the
    // period it is in (a barrier solve's `unit` not yet ended by its result)
    const r0 = c && c.published < 0 ? c.byRound[0].map((r, q) => (r ? q : -1)).filter((q) => q >= 0) : null;
    const who = this.workers.map((_, k) => {
      const h = this.heard[k], u = this.inUnit[k];
      return `w${k} ${h ? `${h.kind} ${s(h.at)}` : "silent"}${u ? ` in r${u.round} p${u.period} ${s(u.at)}` : ""}`;
    });
    const parts = [
      `status ${this.status.slice(0, 120)}`,
      this.statusBefore ? `before ${this.statusBefore.slice(0, 60)}` : "", // i18n-ignore
      c ? `solve v${c.version} came ${s(c.arrivedAt)} ago, sent ${c.sentAt ? s(c.sentAt) : "not yet"}, rounds ${c.roundAt?.length ?? 0}, final ${c.results.every((r) => r) ? "yes" : "no"}` : "no solve", // i18n-ignore
      c ? `lines ${c.svc.lines.length}` : "", // i18n-ignore
      r0 ? `round 0 periods in ${r0.join(",") || "none"} of ${c!.byRound[0].length}` : "", // i18n-ignore
      who.join(", "),
      this.pausedSnap ? `network held while paused ${s(this.pausedAt)}` : "", // i18n-ignore
      `focus ${this.fpool?.alive ? "on" : "off"}`, // i18n-ignore
      this.stuckRestarted ? `restarted when stuck ${s(this.stuckRestarted)} ago` : "", // i18n-ignore
      `shown v${v?.version ?? "-"}${v?.final ? " final" : ""}`, // i18n-ignore
      this.waiting ? `network waiting ${s(this.waiting.arrivedAt)}` : "", // i18n-ignore
      `step ${s(lastStep())} ago`, // i18n-ignore
      oldest === null ? "" : `oldest download ${s(oldest)}`, // i18n-ignore
      open.length ? `downloading ${open.slice(0, 6).join(", ")}` : "",
      `workers ${this.workers.length}, replaced ${this.respawns}`, // i18n-ignore
      this.lastTrouble ? `last trouble ${this.lastTrouble.slice(0, 200)}` : "", // i18n-ignore
      this.failWhy.size ? `zones failed ${[...this.failWhy].slice(0, 4).map(([z, w]) => `${z}: ${w.slice(0, 100)}`).join(" | ")}` : "", // i18n-ignore
    ];
    return parts.filter(Boolean).join("; ");
  }
  /** T-3314: while a network waits to be solved, when the pool last made progress (it came, or the last step),
   * performance.now(); else null (game/figuresStopped.ts) */
  stuckSince(): number | null {
    return this.waiting ? Math.max(this.waiting.arrivedAt, lastStep()) : null;
  }
  /** the oldest pack download in flight's start, performance.now(), or null */
  oldestDownload(): number | null {
    let o: number | null = null;
    for (const at of this.downloadsOpen.values()) if (o === null || at < o) o = at;
    return o;
  }
  private async fetchPackFile(file: string, high: boolean): Promise<Uint8Array> {
    const r = await patientFetch(file === MARKET_ROWS ? file : dataUrl(file), high ? { priority: "high" } as RequestInit : undefined);
    if (!r.ok) throw new Error(`${r.status} ${r.url}`);
    const d: Download = { got: 0, total: Number(r.headers.get("Content-Length")) || 0, at: performance.now() };
    this.downloads.set(file, d);
    if (!r.body) {
      const b = new Uint8Array(await r.arrayBuffer());
      d.got = d.total = b.length;
      d.end = performance.now();
      return b;
    }
    const reader = r.body.getReader(), chunks: Uint8Array[] = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      d.got += value.length;
      markAlive();
      markStep();
      this.progressSoon();
    }
    // a compressed response's length is not its body's: the body is what was downloaded
    d.total = d.got;
    d.end = performance.now();
    if (chunks.length === 1) return chunks[0];
    const out = new Uint8Array(d.got);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  /** Post an `open` to every worker (the last takes the original buffer, the others copies).
   * T-173: until the first worker is ready, the other solving workers' city packs wait (`held`). */
  private post(open: { pack?: { id: CityId; header: string; bin: ArrayBuffer; level?: number }; intercity?: { id: CityId; bytes: Uint8Array }[]; rows?: Uint8Array }) {
    // T-776: the market's rows go with the first market pack, the same way as its bytes
    const rows = open.intercity && !this.rowsSent ? open.rows : undefined;
    if (rows) this.rowsSent = true;
    // T-2045: the zone in view's pack to the focus pool as well, during a game load
    const fp = this.fpool;
    if (open.pack && fp?.alive && open.pack.id === activeCity.peek() && !fp.holds(open.pack.id)) fp.post({ ...open.pack, bin: open.pack.bin.slice(0) });
    // T-3373 B: to the item's holders and the aux worker (every worker unless `?split=1`)
    const to = this.holdersOf(open.pack ? open.pack.id : MARKET);
    const lastSolving = to.filter((i) => i < this.computeN).pop();
    this.workers.forEach((w, i) => {
      if (!to.includes(i)) return;
      const last = i === this.workers.length - 1;
      const pack = open.pack && { ...open.pack, bin: last ? open.pack.bin : open.pack.bin.slice(0) };
      // T-180: a market pack as bytes, each solving worker its own copy (the last solving one the
      // original), decoded there: decoding megabytes and cloning the text six times held the main
      // thread up while the player could already pan. The aux worker never opens one: empty bytes.
      const own = <T extends Uint8Array>(b: T) => (i === this.computeN ? new Uint8Array(0) : i === lastSolving ? b : b.slice());
      const ms = open.intercity?.map((m) => ({ id: m.id, pack: own(m.bytes) })), r = rows && own(rows);
      const dr = r?.byteLength && this.drives ? this.drives.slice() : undefined;
      const cf = r?.byteLength && this.corridorFile ? this.corridorFile.slice() : undefined;
      const msg: ToDemand = { kind: "open", packs: pack ? [pack] : [], ...(ms ? { intercity: ms } : {}), ...(r?.byteLength ? { rows: r } : {}), ...(dr ? { drives: dr } : {}), ...(cf ? { corridors: cf } : {}) };
      const transfer: Transferable[] = pack ? [pack.bin] : [];
      for (const m of ms ?? []) if (m.pack.byteLength) transfer.push(m.pack.buffer as ArrayBuffer);
      if (r?.byteLength) transfer.push(r.buffer as ArrayBuffer);
      if (dr) transfer.push(dr.buffer as ArrayBuffer);
      if (cf) transfer.push(cf.buffer as ArrayBuffer);
      // T-1533: the zone in view's pack as it is sent (a held one may go seconds later), T-2453: or the selected zone's
      const send = () => w.postMessage(pack && this.firstZones().includes(pack.id) ? { ...msg, first: true } : msg, transfer);
      if (pack && i > 0 && i < this.computeN && this.held) this.held.push(send);
      else send();
    });
    if (rows) this.drives = null;
  }
  /** T-173: worker 0 opens a small synthetic everyday layer before it is ready, and the other
   * solving workers get no city pack until then. With five workers opening New York's everyday
   * layer at once from the browser's baseline WebAssembly code, the optimised code came seconds
   * late (7-8 s in every worker instead of 1.5-2 s); compiled code is shared by a page's workers,
   * so after one worker's warm-up they all open on optimised code. Null once released. */
  private held: (() => void)[] | null = [];

  /** `full`: to every worker; `bare` (T-325): to the aux worker alone; `undecided` (T-351): either,
   * by the reach known once it has downloaded. A pack opened bare and reached later is fetched
   * again for every worker (the page keeps no copy). */
  private fetchCity(id: CityId, mode: "full" | "bare" | "undecided") {
    if (mode === "full") { this.fetching.add(id); this.solvingHolders(id); } // (T-3373 B: placed in the order asked)
    else if (mode === "bare") this.fetchingBare.add(id);
    else this.fetchingUndecided.add(id);
    const log: LoadLog = { item: id, requestedAt: performance.now(), ...(mode === "bare" ? { bare: true } : {}) };
    this.start.loads.push(log);
    this.downloadPack(id).then(([header, bin]) => {
      log.downloadedAt = performance.now();
      this.recovered(id);
      // T-351: fetched before the reach was known: bare if it is known now and leaves the zone out,
      // else in every worker (a save's network comes after its viewed pack, which then opens as before)
      let bare = mode === "bare";
      if (mode === "undecided") {
        this.fetchingUndecided.delete(id);
        const reached = this.knownReach();
        bare = reached !== null && !reached.has(id) && !this.fetching.has(id);
        if (bare) { log.bare = true; this.fetchingBare.add(id); }
        else {
          if (reached === null && !this.fetching.has(id)) this.unknownReach.add(id);
          this.fetching.add(id);
        }
      }
      this.bytes.set(id, bin.byteLength);
      const own = bin.byteOffset === 0 && bin.byteLength === bin.buffer.byteLength ? bin.buffer as ArrayBuffer : bin.slice().buffer;
      const pack = { id, header: new TextDecoder().decode(header), bin: own };
      // T-328: hashed before the bytes go to the workers (WebCrypto copies them at the call)
      if (!bare) this.hashPack(id, [header, bin]);
      if (!bare) {
        // T-4051: a zone with no figures shown opens bare in the aux worker first, ahead of its full open there, so
        // the solve that covers it, sent once every worker has it, finds its view and takes the short path
        // (`newZones` made it round by round: 125 s against 35 for Medellin on a big save)
        if (this.bareFirst(id)) {
          const copy = own.slice(0);
          this.fetchingBare.add(id);
          this.start.loads.push({ item: id, requestedAt: log.requestedAt, downloadedAt: log.downloadedAt, bare: true });
          this.workers[this.auxAt].postMessage({ kind: "open", packs: [{ ...pack, bin: copy }], bare: true } satisfies ToDemand, [copy]);
        }
        // T-1223: at the level the fine set gives it now
        const level = this.levelFor(id);
        this.levelPosted.set(id, level);
        this.post({ pack: { ...pack, level } });
        return this.postMarkets();
      }
      const msg: ToDemand = { kind: "open", packs: [pack], bare: true, ...(this.firstZones().includes(id) ? { first: true } : {}) };
      this.workers[this.auxAt].postMessage(msg, [own]);
    // i18n-ignore: a stored name or a developer diagnostic
    }).catch((err) => { this.unreachable(id, id, `city pack ${id}: ${String(err)}`); this.postMarkets(); });
  }

  /** A zone about to open in every worker that an edit's solve would find with no figures shown (`dispatch`). */
  private bareFirst(id: CityId): boolean {
    return !this.hold && !!demandView.peek()?.final && !cityDemandViews.peek().has(id) && !this.bareOpen.has(id) && !this.bareKept.has(id);
  }

  /** T-377: a zone's market pack, to every solving worker. */
  private fetchMarket(id: CityId) {
    const item = marketItem(id);
    this.fetching.add(item);
    const log: LoadLog = { item, requestedAt: performance.now() };
    this.start.loads.push(log);
    this.marketsDownloading.add(id);
    Promise.all([this.download(`${id}.market.json`), this.marketRows()]).then(([bytes, rows]) => {
      log.downloadedAt = performance.now();
      this.recovered(item);
      this.bytes.set(item, bytes.byteLength);
      // T-279: the routed drives change the market's figures as its packs do
      if (this.cacheOn) {
        const h = this.packHashOf(item);
        Promise.all([sha256(bytes as Uint8Array<ArrayBuffer>), this.drivesSha ?? ""]).then((hs) => h.set(hs.join("")), () => h.set("unhashed"));
      }
      this.marketsDownloading.delete(id);
      this.marketsHeld.push({ id, bytes });
      this.heldRows = rows;
      this.postMarkets();
    }).catch((err) => {
      this.marketsDownloading.delete(id);
      this.unreachable(item, id, `market pack ${id}: ${String(err)}`); // i18n-ignore
      this.postMarkets();
    });
  }

  /** T-889: a market pack downloaded while others still download waits for them, so the solving
   * workers compose the market once for them all (1.5-3.5 s a time on a big game's markets). */
  private marketsDownloading = new Set<CityId>();
  private marketsHeld: { id: CityId; bytes: Uint8Array }[] = [];
  private heldRows: Uint8Array | null = null;
  private postMarkets() {
    if (this.marketsDownloading.size || !this.marketsHeld.length) return;
    // T-1533: and after the city packs still downloading or held: a worker composing a big game's
    // market (a minute for China's 370) would hold up the zone in view's pack and its rough solve
    if (this.held?.length || [...this.fetching].some((id) => !isMarket(id) && !this.bytes.has(id))) return;
    // (one closed or given up on meanwhile is not sent)
    const held = this.marketsHeld.filter((m) => this.fetching.has(marketItem(m.id)));
    this.marketsHeld = [];
    if (held.length) this.post({ intercity: held, rows: this.heldRows ?? undefined });
  }

  /** T-776: the world market's rows, downloaded once, beside the first market pack (a fraction of
   * its size, so in before it; preloading them with a zone's city pack cost a new game 0.2 s). */
  private rows: Promise<Uint8Array> | null = null;
  private rowsSent = false;
  /** T-279: the routed drives (market-drives.bin), in with the rows and sent beside them; null
   * where the file is missing, every drive then the rule's. T-2485: dropped once the workers have
   * them (12 MB), downloaded again for a replacement worker; `drivesSha` keys the cache. */
  private drives: Uint8Array | null = null;
  private drivesSha: Promise<string> | null = null;
  /** T-4199: the corridor file (about 420 KB), in with the rows, kept for a replacement worker */
  private corridorFile: Uint8Array | null = null;
  private marketRows(): Promise<Uint8Array> {
    return (this.rows ??= Promise.all([this.download(MARKET_ROWS), fetchMarketDrives(), fetchCorridorFile()])
      .then(([rows, drives, corridors]) => {
        this.drives = drives;
        this.corridorFile = corridors;
        if (drives && this.cacheOn) this.drivesSha = sha256(drives as Uint8Array<ArrayBuffer>);
        return rows;
      })
      .catch((err) => { this.rows = null; throw err; }));
  }

  // --- T-484: a pack that could not be downloaded --------------------------------------------------

  /** Items (a city id or "market:<id>") whose download failed, with when to try again. Demand
   * carries on without them: they are left out of what is needed, waited for and solved. */
  private broken = new Map<string, { tries: number; timer: number }>();

  /** `item` (of zone `zone`) could not be downloaded even after its retries: leave it out, say so for
   * its zone, and try again later (BROKEN_RETRY_MS) if it is still needed then. */
  private unreachable(item: string, zone: CityId, why: string, retry = true) {
    if (this.failed) return;
    console.warn("demand:", why);
    this.fetching.delete(item);
    this.fetchingBare.delete(item as CityId);
    this.fetchingUndecided.delete(item as CityId);
    this.prefetched.delete(item as CityId);
    this.releveling.delete(item as CityId);
    this.onOpen = this.onOpen.filter((w) => (w.item === item ? (w.f(false), false) : true));
    const tries = (this.broken.get(item)?.tries ?? 0) + 1;
    const wait = BROKEN_RETRY_MS[Math.min(tries, BROKEN_RETRY_MS.length) - 1];
    const timer = retry ? setTimeout(() => {
      this.broken.delete(item);
      this.ensure();
    }, wait) as unknown as number : 0;
    this.broken.set(item, { tries, timer });
    const failures = new Map(zoneFailures.peek());
    failures.set(zone, t("demandClient.zoneFailed", { why, retry }));
    zoneFailures.value = failures;
    this.failWhy.set(zone, why);
    this.ensure();
    // a game load's hold may have been waiting for this zone alone
    this.checkHold();
  }

  /** T-492: items a worker could not open, with how many times: the zone stays flagged until the
   * item opens. The first failure is tried again once (the pack may have been rewritten while it
   * downloaded); a second one is final, since a pack that downloads whole and does not open twice is
   * broken, not caught mid-write. */
  private openFails = new Map<string, number>();

  /** A worker could not open a pack of the `open` it was sent: it is left out as a pack that could not
   * be downloaded is. Every worker is told to drop it (another may have opened it), and the first
   * worker's word is taken for all (the same pack fails the same way in each). */
  private openRejected(r: OpenRejected, bare: boolean) {
    const item = r.market ? marketItem(r.id) : r.id;
    if (this.broken.has(item)) return;
    const fails = (this.openFails.get(item) ?? 0) + 1;
    this.openFails.set(item, fails);
    if (!bare) {
      for (const w of this.workers) w.postMessage({ kind: "close", ids: r.market ? [] : [r.id], ...(r.market ? { markets: [r.id] } : {}) } satisfies ToDemand);
      this.opened.delete(item);
      this.unack(item);
      this.packAcks.delete(item);
    }
    this.unreachable(item, r.id, `${r.market ? "market pack" : "city pack"} ${r.id} would not open: ${r.error}`, fails < 2); // i18n-ignore
  }

  /** `item` downloaded: whatever failed before for it is over, unless it also did not open. */
  private recovered(item: string) {
    const b = this.broken.get(item);
    if (b) clearTimeout(b.timer);
    this.broken.delete(item);
    if (this.openFails.has(item)) return;
    const zone = (isMarket(item) ? marketZone(item) : item) as CityId;
    if (!zoneFailures.peek().has(zone) || this.broken.has(zone) || this.broken.has(marketItem(zone))) return;
    const failures = new Map(zoneFailures.peek());
    failures.delete(zone);
    zoneFailures.value = failures;
    this.failWhy.delete(zone);
  }

  /** T-3628: each flagged zone's raw reason, for a feedback report */
  private failWhy = new Map<CityId, string>();

  /** T-3628: the page came back to the foreground or the browser went online: every pack waiting out its
   * retry delay is tried now, once. One already downloading is not in `broken`, so it is left alone. */
  retryBroken() {
    let any = false;
    for (const [item, b] of this.broken) {
      if (!b.timer) continue; // a pack that would not open twice is final
      clearTimeout(b.timer);
      this.broken.delete(item);
      any = true;
    }
    if (any) this.ensure();
  }

  /** `ids` without the zones whose pack is waiting to be tried again. */
  private withoutBroken(ids: Set<CityId>): Set<CityId> {
    for (const id of ids) if (this.broken.has(id)) ids.delete(id);
    return ids;
  }

  /** A worker opened `item`: once all have, waiting solves and queries may go. */
  private openedBy(item: string, i: number) {
    if (this.broken.has(item)) return; // T-492: another worker could not open it
    const by = (this.ackedBy[i] ??= new Set());
    if (by.has(item)) return;
    by.add(item);
    if (!this.holds(i, item)) return; // (T-3373 B: a worker that no longer holds it, after a shrink)
    const n = (this.acks.get(item) ?? 0) + 1;
    this.acks.set(item, n);
    if (n >= this.holdersOf(item).length) this.openEverywhere(item);
  }

  /** Every worker that should holds `item` (T-3373 B: its holders and the aux worker under `?split=1`). */
  private openEverywhere(item: string) {
    this.opened.add(item);
    // T-4051: a market pack opened after the load, from its download to every worker holding it (on a big save
    // the workers finish the unit in hand and compose the market first, far past OPEN_MARKET_MS), so the next
    // zone's bar expects as long
    if (isMarket(item) && !this.hold && startup.peek() !== "loading") {
      const log = lastLoad(this.start.loads, (l) => l.item === item && !l.bare);
      if (log?.downloadedAt !== undefined && log.openedAt === undefined) this.marketOpenMs = performance.now() - log.downloadedAt;
    }
    if (this.releveling.delete(item as CityId)) this.lodShown();
    if (this.openFails.delete(item)) this.recovered(item);
    this.logLoad(item, false, "openedAt");
    this.settleWaiters();
  }

  /** T-4051: the last market batch's time from its download to open in every worker, after the load (none before) */
  private marketOpenMs = 0;

  /** T-325: the aux worker opened zones bare: their figures and city views. */
  private bareOpened(figures: Record<CityId, BareFigures>) {
    for (const [id, f] of Object.entries(figures)) {
      this.bareOpen.set(id, f);
      this.bareKept.delete(id);
      if (this.openFails.delete(id)) this.recovered(id);
      this.logLoad(id, true, "openedAt");
    }
    this.addBareViews();
    this.settleWaiters();
    if (trackZones.peek().built.size) this.ensure(); // T-1047: the next zone track crosses
  }

  /** Resolve `whenOpen` waits for items now open (bare ones need only the aux worker). */
  private settleWaiters() {
    const all = this.ready === this.workers.length;
    this.onOpen = this.onOpen.filter((w) => ((this.opened.has(w.item) && all) || this.bareOpen.has(w.item) ? (w.f(true), false) : true));
  }

  /** Send the waiting service when everything it needs is open; or solve again for a city opened
   * since the newest solve (the viewed city after a pan). */
  private flush(force = false) {
    if (this.failed) return;
    if (this.idle()) return this.settleIdle();
    if (!force && this.firm()) {
      const s = this.cur;
      if (this.waiting && s) {
        s.heldFrom ??= performance.now();
        // T-3616: held only until the local result, likely: no wait for the whole solve to count, no zone solved aside
        if (this.quickHold(s)) {
          if (!s.chainTimer) {
            s.chainTimer = true;
            const end = Math.min(s.sentAt + HOLD_QUIET_MS, (s.chainFrom ?? s.sentAt) + LOCAL_CHAIN_MS);
            setTimeout(() => { if (this.cur === s) this.flush(); }, Math.max(0, end - performance.now()) + 50);
          }
          return;
        }
        if (s.refresh) this.showRefresh(s);
        else if (editHeld.peek()?.sentAt !== s.sentAt) editHeld.value = { sentAt: s.sentAt };
        this.asideSoon(); // T-4220: held behind a whole solve
      }
      this.refineAside();
      return;
    }
    this.closeStrays();
    // T-4199: nothing is sent while a game load's network is on its way: the network in hand or waiting is the last
    // game's (a solve of it with the packs opened ahead of the load ended the load's focus pool with its final)
    if (gameLoadArmed()) return;
    // T-325: a solve waits for, and covers, the cities stations reach and those opened in every
    // worker; a zone opened bare is never waited for and starts no solve
    if (this.waiting) {
      const need = this.solveNeed(this.waiting.svc);
      if (need && this.allOpen(need)) {
        const w = this.waiting;
        this.waiting = null;
        this.dispatch(w.svc, w.arrivedAt);
      } else this.roughSoon(this.waiting.svc, need);
      return;
    }
    const s = this.cur;
    const need = s && this.solveNeed(s.svc);
    if (!s || !need || !this.allOpen(need)) return;
    // a city opened since the newest solve (also one that finished loading after the view moved
    // away again: it is open, so solve it rather than leave its load unfinished)
    // (T-1223: or opened again at another level since)
    if (!this.covers(s, need)) this.dispatch(s.svc, performance.now());
  }

  /** T-4055: `s` covers every pack a solve of its network would now (`need`, the markets open and wanted), each at
   * the level it is open at, and none of them is opening (reached, wanted, or opening again at another level since
   * `s` was sent). Else a solve follows once they are open (`flush`), and `s`'s final is not its network's. */
  private covers(s: Solve, need: Set<CityId>): boolean {
    const coarse = new Set(s.coarse ?? []);
    return [...need].every((id) => this.opened.has(id) && s.cities.includes(id) && coarse.has(id) === (this.levelPosted.get(id) === LOD_COARSE))
      && this.marketWanted.every((id) => this.opened.has(marketItem(id)) || !this.fetching.has(marketItem(id)))
      && [...this.opened].every((item) => !isMarket(item) || s.markets.includes(marketZone(item)));
  }

  /**
   * T-2579: zones opened in every worker before the reach was known (`fetchCity`: the zone in view as
   * the page started, the last game's place while a save loads). In a game load's hold, once its network's
   * reach is known, one it neither reaches nor views closes before the solve covers it: the load's
   * figures, and the cache written from them, hold no zone that another load of the same network
   * would not open.
   */
  private unknownReach = new Set<CityId>();
  private closeStrays() {
    if (!this.hold || !this.unknownReach.size) return;
    const reached = this.knownReach();
    if (!reached) return;
    for (const id of this.unknownReach) if (reached.has(id) || !this.fetching.has(id)) this.unknownReach.delete(id);
    const viewed = this.viewedSet();
    const stray = [...this.unknownReach].filter((id) => !viewed.has(id) && this.opened.has(id) && !this.releveling.has(id));
    if (stray.length) this.closeFull(stray, []);
  }

  /**
   * T-1734: an edit's solve runs to its end: from when it is sent until its final lands, a newer
   * network (an edit, a zone opened or at another level) waits, and is sent the moment the final
   * is shown, solved locally on top of it. The one gap is the moment after the local result shows,
   * until `whole` goes (`EDIT_SETTLE_MS`), so an edit right after another (a line, then its trains)
   * is solved locally too. A network held back since before that goes in the gap too while the edits since
   * a whole solve last started began at most `LOCAL_CHAIN_MS` before the local result (T-3616), and waits for
   * the final after that, or edits faster than a slow local solve would never let a whole solve start.
   */
  private firm(): boolean {
    const s = this.cur;
    if (!s || s.barrier !== false || s.rough || s.published >= ROUNDS || perfOff("firm")) return false;
    if (this.wholeRuns(s)) return !this.restarts(s);
    if (s.localShownAt === undefined) return true;
    if (s.heldFrom === undefined || s.heldFrom > s.localShownAt) return false;
    return s.localShownAt - (s.chainFrom ?? s.sentAt) > LOCAL_CHAIN_MS;
  }

  /** T-3925: `s`'s whole solve has started: `whole` went, or no worker has a local stage (it starts at once) */
  private wholeRuns(s: Solve): boolean {
    return !!s.wholeSent || Array.from({ length: this.computeN }, (_, k) => s.localPlans?.[k]).every((v) => v === 0);
  }

  /** T-3925: a newer network (an edit, not a zone opened or at another level) stops `s`'s whole solve now
   * (`RESTART_SHARE`); while the time left can't be told it waits as before */
  private restarts(s: Solve): boolean {
    const w = this.waiting;
    if (!w || w.svc === s.svc || perfOff("restart")) return false;
    const now = performance.now(), left = unitsLeft(s, this.computeN, now);
    if (left === null) return false;
    const ran = now - s.sentAt, all = ran + left;
    return ran < RESTART_SHARE * all && now - (s.restartFrom ?? s.sentAt) < all;
  }

  /** T-3616: a network held now will likely go at `s`'s local result (`firm`): its local stage runs, sent at most
   * `HOLD_QUIET_MS` ago, and within the chain's `LOCAL_CHAIN_MS` */
  private quickHold(s: Solve): boolean {
    if (s.wholeSent || s.localShownAt !== undefined) return false;
    // (no worker solves locally: it is solved whole at once; the plans come within a second or so)
    const plans = Array.from({ length: this.computeN }, (_, k) => s.localPlans?.[k]);
    if (plans.every((v) => v === 0)) return false;
    const now = performance.now();
    return now - s.sentAt <= HOLD_QUIET_MS && now - (s.chainFrom ?? s.sentAt) <= LOCAL_CHAIN_MS;
  }

  /**
   * T-1533: while a whole solve waits for packs still downloading or opening (a big game's load, a
   * zone just reached), the zone in view, once open in every worker, is solved alone with no market
   * (`only`): its figures and bubbles show as an estimate within seconds, and the whole solve's
   * replace them. Not when the zone in view already has figures (a cached result, an earlier solve).
   */
  private roughSoon(svc: Service, need: Set<CityId> | null) {
    const id = activeCity.peek();
    // (T-2045: during a game load the focus pool solves the zone in view, to its last round)
    // (T-3373 B: under `?split=1` the zone in view's holder solves it alone; `postSolve`)
    if (this.fpool?.alive || perfOff("rough") || !need?.has(id) || !this.viewedSet().has(id) || this.ready !== this.workers.length || !this.opened.has(id)) return;
    if (this.rough?.svc === svc && this.rough.id === id) return;
    // a zone with figures is solved rough only for lines new since them, which take its free-flow
    // round's figures as a whole solve's would (T-324); its other figures stay
    const shown = cityDemandViews.peek().has(id);
    if (shown && !this.hasNewLines(svc)) return;
    this.rough = { svc, id };
    const { net, layout, ids } = packService(svc);
    const version = ++this.version;
    this.keepShownSums(null);
    const now = performance.now();
    // T-2793: its figures are this zone's riders alone: over a final result they are for the zone's own view, no line's
    const onFinal = !!demandView.peek()?.final;
    this.cur = { version, svc, layout, results: PERIODS.map(() => null), byRound: Array.from({ length: ROUNDS + 1 }, () => PERIODS.map(() => null)), published: -1, arrivedAt: now, sentAt: now, periodMs: [], setupMs: [], memBytes: [], cities: [id], markets: [], coarse: this.coarseOf([id]), showEstimates: !shown && !onFinal, newZones: !shown && onFinal, rough: true };
    this.log.push(this.cur);
    if (this.log.length > 50) this.log.shift();
    demandUpdating.value = true;
    demandScope.value = 0;
    const msg: ToDemand = { kind: "solve", version, net, rounds: ROUNDS, barrier: true, first: id, only: [id], ids, ...(perfOff("estimate") ? { exact0: true } : {}) };
    this.postSolve(msg);
  }
  private rough: { svc: Service; id: CityId } | null = null;
  /** T-2658: the world version each network was last given at */
  private netWorld = new WeakMap<Service, number>();
  /** `svc` has lines the shown result has no figures for (new, or their stops changed) */
  private hasNewLines(svc: Service): boolean {
    const v = demandView.peek()?.version;
    const base = this.log.find((x) => x.version === v);
    if (!base) return false;
    const had = new Map(base.layout.map((l) => [l.id, l.stops.join(",")]));
    return svc.lines.some((l) => had.get(l.id) !== l.stops.join(","));
  }

  /** Resolves when `item` is open in every worker, or bare in the aux worker (false if the pool
   * failed). */
  private whenOpen(item: string): Promise<boolean> {
    if (this.failed) return Promise.resolve(false);
    if ((this.opened.has(item) && this.ready === this.workers.length) || this.bareOpen.has(item as CityId)) return Promise.resolve(true);
    return new Promise((f) => this.onOpen.push({ item, f }));
  }

  /** T-325: whether a load needs a solve to finish: at startup, in a game load's hold, or when a
   * city of it must be covered by the solves (not a zone opened bare) and has no figures on the map yet. */
  private jobNeedsSolve(job: NonNullable<DemandClient["job"]>): boolean {
    // (T-2552: and a game load whose network has not arrived: the network in hand is the last game's)
    if (job.startup || this.hold || (job.wait !== null && gameLoadArmed())) return true;
    const need = this.solveNeed(this.latest());
    // T-4359: a city with figures shown (its bare ones: a zone opens bare first) is loaded once its packs are open. The solve
    // that takes it in is then an edit's, which publishes nothing before its final, so waiting on it would keep "Loading
    // data" up, its bar full, for a whole re-solve; the status tag says that wait is calculating trips.
    const shown = cityDemandViews.peek();
    return !need || job.cities.some((id) => need.has(id) && !shown.has(id));
  }

  private newJob(atStartup: boolean): NonNullable<DemandClient["job"]> {
    const now = performance.now(), w = gameWait();
    const held = !!this.hold || gameLoadArmed();
    return { items: [], cities: [], startup: atStartup, best: 0, since: held && w.since > 0 ? Math.min(w.since, now) : now, wait: held ? w.seq : null };
  }

  private progressSoon() {
    if (!this.progressTimer) this.progressTimer = setTimeout(() => { this.progressTimer = 0; this.progress(); }, 100) as unknown as number;
  }

  private progressAt = -Infinity;
  /** At most every PROGRESS_GAP_MS: a big load calls this on every message, about 830 times in 30 s on Japan
   * (T-1172), and a call that comes too soon runs once the gap is over. */
  private progress() {
    const t0 = performance.now();
    if (t0 - this.progressAt < PROGRESS_GAP_MS) {
      if (!this.progressTimer) this.progressTimer = setTimeout(() => { this.progressTimer = 0; this.progress(); }, PROGRESS_GAP_MS - (t0 - this.progressAt) + 1) as unknown as number;
      return;
    }
    this.progressAt = t0;
    this.jobProgress();
    this.zoneProgress();
  }

  /** T-343, T-2552: each loading zone's bar as shown so far (its best and place) */
  private zonePlaces = new Map<CityId, BarPlace>();
  /** T-421: a game load's zones still without their final after its card went at the view's */
  private unsettled: Set<CityId> | null = null;
  /** T-421: a startup solve released at the view's final, remembered (LAST_SOLVE) at its own */
  private timeRest?: { version: number; stops: number; finalMs?: number };

  /** T-343: every zone fetched (in every worker or bare) whose figures have not arrived, with its
   * share done (game/loading.ts `zoneLoads`): the load in hand's for its zones, else its own
   * download and opening, each weighted by its expected time as the load's are. */
  private zoneProgress() {
    const shown = cityDemandViews.peek(), now = performance.now();
    const next = new Map<CityId, number>();
    const steps = new Map<CityId, ZoneStep>();
    const load = loadProgress.peek();
    const s = this.cur;
    // T-2524: the time a whole solve with the barrier still needs (a load's, or a cached result's network
    // solved again), for the zones it holds up and the note of a click waiting on it
    const st = s && s.published < ROUNDS && !this.failed && (this.unsettled?.size || s.barrier)
      ? solveStage(s.sentAt, s.roundAt, s.published, s.byRound[s.published + 1]?.filter(Boolean).length ?? 0, s.roundFirstAt, ROUNDS, solveEstimate(stopCount(s.svc), "all"), loadPace().later, now, PERIODS.length, solveWaves, setupOf(s))
      : null;
    const was = solveLeft.peek();
    const units = s && !this.failed ? unitsBound(s, this.computeN, now) : null;
    const byUnits = st ? units?.ms ?? null : null;
    // T-4264: the running solve's share by its units, where they bound its time left (null otherwise): a zone's bar
    // waiting on it passes half only on this
    const unitsShare = s && units?.bounded ? Math.max(1, now - s.sentAt) / (Math.max(1, now - s.sentAt) + units.ms) : null;
    const left = st && s?.barrier && !s.rough ? { version: s.version, leftMs: nextLeft(was?.version === s.version ? was.leftMs : null, byUnits ?? st.left, was ? now - was.at : 0), at: now, bounded: byUnits !== null && !!units?.bounded } : null;
    // (an estimate that only counts down is not worth a render: the bars and the note count down between)
    const ends = (a: { leftMs: number | null; at: number }, b: { leftMs: number | null; at: number }) => (a.leftMs === null) === (b.leftMs === null) && Math.abs((a.at + (a.leftMs ?? 0)) - (b.at + (b.leftMs ?? 0))) < 1000;
    if (!left || !was || left.version !== was.version || !ends(left, was) || left.bounded !== was.bounded) solveLeft.value = left;
    const sl = solveLeft.peek();
    const solving: ZoneStep = { step: STAGE_TEXT.demand, leftMs: sl?.leftMs ?? null, at: sl?.at ?? now };
    let moving = false;
    // T-2552: each bar keeps its place as the card's does (`barPlace`); none is kept while the card is up, which
    // hides the bars, so a zone's bar starts from its own share when it shows
    const cardUp = !!clock?.held.peek();
    const places = new Map<CityId, BarPlace>();
    const placed = (id: CityId, f: number, bounded: number | null) => {
      if (cardUp) return barPlace({ best: 0 }, f, bounded);
      const p = this.zonePlaces.get(id) ?? { best: 0 };
      places.set(id, p);
      return barPlace(p, f, bounded);
    };
    const sn = this.solveNeed(this.latest());
    for (const id of [...this.fetching, ...this.fetchingBare, ...this.fetchingUndecided]) {
      if (isMarket(id) || shown.has(id) || next.has(id) || this.unsettled?.has(id) || this.failed) continue;
      let f: number, bounded: number | null = null;
      const ofJob = !!this.job?.cities.includes(id);
      if (ofJob) {
        f = this.job!.best;
        steps.set(id, { step: load?.step ?? STAGE_TEXT.data, leftMs: load?.leftMs ?? null, at: now });
      } else {
        // (a zone a solve covers opens in every solving worker: their acks are its real progress)
        const toSolve = !!sn?.has(id);
        const covered = !!s && !s.rough && s.cities.includes(id);
        // T-4359: once open where the solves need it, it waits on the calculation, never on its packs (an edit's solve has no
        // `st`; its time left is its workers' units)
        const calc = (toSolve && this.opened.has(id)) || (this.isOpen(id) && !!st && covered);
        steps.set(id, !calc ? { step: STAGE_TEXT.data, leftMs: null, at: now } : covered && st ? solving : { step: STAGE_TEXT.demand, leftMs: covered ? units?.ms ?? null : null, at: now });
        let got = 0, total = 0, at = Infinity;
        for (const file of [`${id}.json`, `${id}.bin`]) {
          const d = this.downloads.get(file);
          if (d) { got += d.got; total += d.total || d.got; at = Math.min(at, d.at); }
        }
        const bytes = this.bytes.get(id);
        const dl = bytes !== undefined ? 1 : total > 0 ? Math.min(0.99, got / total) : 0;
        const dlMs = at === Infinity ? 300 : dl >= 1 || got === 0 ? Math.max(300, now - at) : (now - at) / dl;
        const openMs = Math.max(300, (bytes ?? (total || GUESS_BYTES)) / 1e6 * OPEN_MS_PER_MB.city);
        const log = lastLoad(this.start.loads, (l) => l.item === id && l.downloadedAt !== undefined && !l.openedAt);
        const acked = toSolve ? Math.min(1, 0.9 * (this.packAcks.get(id) ?? 0) / this.solvingHolders(id).length) : 0;
        const open = this.isOpen(id) ? 1 : Math.max(acked, log ? Math.min(STEP_CAP, (now - log.downloadedAt!) / openMs) : 0);
        // T-2552: and it waits for that solve: the one running, to its final with the barrier and to its first
        // round without, else the next, expected as the solves before it took
        let solveMs = 0, solve = 0;
        if (toSolve) {
          if (covered && s) {
            const zs = st ?? solveStage(s.sentAt, s.roundAt, s.published, s.byRound[s.published + 1]?.filter(Boolean).length ?? 0, s.roundFirstAt, READY_ROUND, solveEstimate(stopCount(s.svc)), loadPace().later, now, PERIODS.length, solveWaves, setupOf(s));
            solveMs = zs.ms;
            solve = zs.share;
            if (open >= 1) bounded = unitsShare;
          } else {
            const svc = this.latest();
            solveMs = solveEstimate(svc ? stopCount(svc) : null);
          }
          moving = true;
        }
        f = zoneShare({ dlMs, dl, openMs, open, solveMs, solve });
        if (!this.isOpen(id)) moving = true;
      }
      // (a zone of the load in hand has the card's bar, placed already)
      next.set(id, ofJob ? f : placed(id, f, bounded));
    }
    // T-421: a game load's zones still solving after its card went at the view's final, until theirs
    // lands (through any newer solve: they have no final figures yet)
    if (s && st && this.unsettled?.size) {
      for (const id of this.unsettled) {
        if (next.has(id)) continue;
        next.set(id, placed(id, st.share, unitsShare));
        steps.set(id, solving);
        moving = true;
      }
    }
    this.zonePlaces = places;
    const prev = zoneLoads.peek();
    if (next.size !== prev.size || [...next].some(([id, f]) => prev.get(id) !== f)) zoneLoads.value = next;
    const prevSteps = zoneSteps.peek();
    const same = (a: ZoneStep, b?: ZoneStep) => !!b && a.step === b.step && ends(a, b);
    if (steps.size !== prevSteps.size || [...steps].some(([id, z]) => !same(z, prevSteps.get(id)))) zoneSteps.value = steps;
    // a bare zone's load (and, T-421, a released solve's zones) moves with time; nothing else ticks for it
    if (moving && !this.progressTick) this.progressSoon();
  }

  /** the last bar's parts: each step's expected ms and share done (window.twDemand.client.barParts with ?debug=1) */
  barParts?: Record<string, number>;

  /** The loading progress: each stage with its share of the bar, T-1195 (game/loadBar.ts, game/loading.ts). */
  private jobProgress() {
    if (this.failed) return;
    const known = this.needed(this.latest());
    // T-889, T-1194: a game load whose network has not arrived has a load in hand (its first stage),
    // whatever the camera passes over on its way to the save's view
    const preNet = gameLoadArmed() && !choosing.peek();
    if (known?.size === 0 && !preNet) {
      // T-318: nothing to load or solve (dead space); the load in hand, if any, finishes unseen
      if (this.job) this.endJob();
      loadProgress.value = null;
      return;
    }
    const need = known ?? this.viewedSet();
    const viewed = activeCity.peek();
    // T-325: a city the solves must cover is loaded when open in every worker; another, when open bare
    const sn = this.solveNeed(this.latest());
    // (T-484: a pack waiting to be tried again counts as done: the load does not wait for it)
    const loaded = (item: string) => this.broken.has(item) || this.releveling.has(item as CityId) || (isMarket(item) || sn === null || sn.has(item as CityId) ? this.opened.has(item) : this.isOpen(item));
    const marketsMissing = this.marketWanted.map(marketItem).filter((item) => !this.opened.has(item));
    // T-328: a zone the shown cached result covers has its figures; it loads unseen (T-347: not
    // while a game load's card is up, which counts everything to the fresh final)
    const cached = this.cacheShown() && !this.hold ? this.cachedCities : null;
    // T-347: a game load's solve is done at its final round; any other load's, at its estimate
    const doneRound = this.hold ? HOLD_ROUND : READY_ROUND;
    const missing = [...need].filter((id) => !loaded(id) && !cached?.has(id)).sort((a, b) => (a === viewed ? -1 : b === viewed ? 1 : 0));
    const atStartup = startup.peek() === "loading";
    const s = this.cur;
    if (!this.job) {
      // T-243: a loaded game whose cities are all open (a browser slot) still waits for its solve
      const heldSolve = !!this.hold && this.hold.version !== undefined && s?.version === this.hold.version && s.published < doneRound;
      if (!missing.length && !(atStartup && marketsMissing.length) && !heldSolve && !preNet) {
        if (!atStartup) loadProgress.value = null;
        return;
      }
      this.job = { ...this.newJob(atStartup), version: heldSolve && !missing.length ? s!.version : undefined };
    }
    const job = this.job;
    for (const id of missing) if (!job.items.includes(id)) { job.items.push(id); job.cities.push(id); }
    for (const item of marketsMissing) if (!job.items.includes(item)) job.items.push(item);
    if (atStartup && need.has(viewed) && !job.cities.includes(viewed)) job.cities.unshift(viewed);
    const needsSolve = this.jobNeedsSolve(job);
    // T-1195: the game stage ends when a game load's network reaches the pool, the data stage when every
    // item of the load is open
    const netIn = job.wait === null || !preNet;
    if (netIn) job.netAt ??= performance.now();
    job.dataAt = netIn && job.items.every(loaded) ? job.dataAt ?? performance.now() : undefined;
    // (T-402: or, when the load's zones are all in the cluster solved ahead, that cluster's round;
    // T-421: a game load's, once the zone in view has its final)
    const ahead = s?.ahead && job.cities.every((id) => s.ahead!.cities.includes(id)) ? s.clusterRes?.get(s.ahead.key)?.published ?? -1 : -1;
    const ofJob = !!s && job.version !== undefined && s.version === job.version;
    const solved = !preNet && (!needsSolve || (ofJob && (Math.max(s.published, ahead) >= doneRound || (!!this.hold && this.viewHasFinal(s)))));
    if (solved && job.items.every(loaded)) {
      // T-884: a game load's card goes with this; its wait is remembered for the game's next load (a new
      // game's under "new", for the next new game), T-1195: each stage's time while it was in hand
      const w = gameWait(), end = performance.now();
      // T-3881: only a load that ran a solve of its own says how long one takes. One that ended with a cached result's
      // card, or with zones that needed none, has a demand stage of seconds, and the next load of the game (a cold
      // one) would give its solve that small a share of the bar and sit at the end of it for minutes.
      if (job.wait === w.seq && s && job.solveRounds) {
        const net = job.netAt ?? job.since, data = Math.max(net, job.dataAt ?? end);
        rememberWait(w.key, { ms: Math.round(end - job.since), stops: stopCount(s.svc), stages: { game: Math.round(net - job.since), data: Math.round(data - net), demand: Math.round(end - data) } });
        const pace = job.pace ?? loadPace();
        const r = job.solveRounds;
        rememberPace({
          data: job.modelData && data - net > 500 ? (data - net) / job.modelData : pace.data,
          game: net - job.since > 200 ? (net - job.since) / GAME_MS : pace.game,
          solve: job.modelSolve && end - data > PACE_SOLVE_MS ? (end - data) / job.modelSolve : pace.solve,
          later: r?.measured && r.first > 0 && end - data > PACE_SOLVE_MS ? r.rest / r.first : pace.later,
        });
      }
      // remembered for the next load's estimate (T-243): a startup's solve, so it describes the
      // game a reload opens; T-347, T-421: and the time to the card going, which its bar measures
      // (not a bare start's, T-351: its solve covers no zone and says nothing of the next game's)
      if (job.startup && s && s.cities.length) {
        const stops = stopCount(s.svc);
        // (T-2459: a card that went at the focus pool's final of the zone in view goes now)
        const at = s.refinedAt ?? (this.selectedAhead(s) ? performance.now() : s.ahead && s.clusterRes?.get(s.ahead.key)?.at);
        const finalMs = at !== undefined ? Math.round(at - s.sentAt) : undefined;
        // T-421: a solve released at the view's final is remembered again at its own final (`publish`);
        // T-884: and now, since a later solve (a zone opened after the card) may take its place first
        const first = s.firstAt !== undefined ? s.firstAt - s.sentAt : finalMs !== undefined ? finalMs / FINAL_FACTOR : undefined;
        if (first !== undefined) rememberSolve(Math.round(first), stops, finalMs, s.refinedAt !== undefined ? finalMs : undefined);
        if (s.refinedAt === undefined) this.timeRest = { version: s.version, stops, finalMs };
      }
      this.endJob();
      loadProgress.value = null;
      return;
    }
    if (!this.progressTick) this.progressTick = setInterval(() => this.progress(), 100) as unknown as number;
    const now = performance.now();
    const within = (since: number | undefined, expected: number) => timeShare(since, expected, now);
    // Downloading: bytes; expected to take as long as its rate so far says.
    let got = 0, total = 0, dlStart = Infinity, dlEnd = 0;
    // Opening: each item in every worker, expected from its size; worker acks are its real progress.
    let openMs = 0, openDone = 0, openStart = Infinity;
    const markets = job.items.filter(isMarket).length;
    for (const item of job.items) {
      let size = 0;
      for (const f of isMarket(item) ? [`${marketZone(item)}.market.json`] : [`${item}.json`, `${item}.bin`]) {
        const d = this.downloads.get(f);
        if (d) { got += Math.min(d.got, d.total || d.got); total += d.total || d.got; dlStart = Math.min(dlStart, d.at); dlEnd = Math.max(dlEnd, d.end ?? now); size += d.total || d.got; }
      }
      const mb = (this.bytes.get(item) ?? (size || GUESS_BYTES)) / 1e6;
      const ms = isMarket(item) ? Math.max(OPEN_MARKET_MS, this.marketOpenMs) / markets + mb * OPEN_MS_PER_MB.market : OPEN_CITY_MS + mb * OPEN_MS_PER_MB.city;
      openMs += ms;
      // T-1195: by the packs in at the solving workers (the aux worker opens far faster); a market pack
      // in counts less, as the market is composed after a batch's last
      openDone += loaded(item) ? ms : ms * Math.min(1, (isMarket(item) ? 0.3 : 0.9) * (this.packAcks.get(item) ?? 0) / this.solvingHolders(item).length);
      const log = lastLoad(this.start.loads, (l) => l.item === item);
      if (log?.downloadedAt !== undefined) openStart = Math.min(openStart, log.downloadedAt);
    }
    const downloaded = job.items.every((i) => this.bytes.has(i) || loaded(i));
    const dl = downloaded ? 1 : total > 0 ? Math.min(0.99, got / total) : 0;
    // (T-793: a finished download stays as long as it took, rather than growing with the clock)
    const dlElapsed = dlStart < Infinity ? (downloaded && dlEnd > 0 ? dlEnd : now) - dlStart : 0;
    const dlMs = downloaded ? dlElapsed : got > 0 && total > got ? dlElapsed * total / got : 300;
    const open = openMs > 0 ? Math.max(openDone / openMs, job.items.every(loaded) ? 1 : within(openStart < Infinity ? openStart : undefined, openMs)) : 1;
    // Solving: period results landed, and time against the expected solve (none for a load of
    // zones opened bare, T-325).
    const svc = this.latest();
    // (T-351: a solve covering no zone, the market alone, is short)
    const stops = svc ? stopCount(svc) : null;
    // T-1195: what this browser measured on the slot's last load, while the network is about its size
    const rec = job.wait !== null ? lastWait(gameWait().key) : null;
    const was = rec?.stages && sameSize(rec, stops) ? rec.stages : null;
    job.pace ??= loadPace();
    // T-1479 (T-890): a game's solve with nothing remembered of it, from the solve model fitted on
    // measured loads, scaled by how this browser's last solve compared with that model
    const gameLoad = !!this.hold || gameLoadArmed();
    if (netIn && stops !== null && sn) job.modelSolve ??= solveModelMs(stops, sn.size);
    const prior = !needsSolve ? 0 : was ? was.demand : sn?.size === 0 ? SOLVE_MS.base : gameLoad && job.modelSolve !== undefined ? job.modelSolve * job.pace.solve : solveEstimate(stops, gameLoad);
    let solveMs = prior, solve = 0, solveLeft = prior;
    // T-4347: the time left shows from the start, on `prior` until the workers' units tell (T-4209 held it back: a guess of
    // 3 min became 8 on asianita; a rough figure that is then corrected beats a blank one, and the bar stays under half
    // while the time is a guess, `bounded`)
    // T-4264: the solve's share by its units, once they bound its time left and the stages before it are done (null
    // otherwise): the bar passes half only on this
    let bounded: number | null = null;
    if (needsSolve && s && job.version !== undefined && s.version === job.version) {
      // T-793: a game load's zone in view is final with its cluster solved ahead (T-402): that cluster's rounds
      const a = this.hold && s.ahead && [...this.viewedSet()].every((id) => s.ahead!.cities.includes(id)) ? s.clusterRes?.get(s.ahead.key) : undefined;
      const published = (a ?? s).published, byRound = (a ?? s).byRound;
      const landed = byRound[published + 1]?.filter(Boolean).length ?? 0;
      // (a cluster solved ahead keeps no landing times: seen here, to 0.1 s)
      if (a && landed > 0) (job.firstLand ??= [])[published + 1] ??= now;
      const st = solveStage(s.sentAt, (a ?? s).roundAt, published, landed, a ? job.firstLand : s.roundFirstAt, doneRound, prior, job.pace.later, now, PERIODS.length, solveWaves, a ? 0 : setupOf(s));
      solveMs = st.ms;
      solve = st.share;
      solveLeft = st.left;
      // T-3876: by the workers' units where they can tell, as an edit's solve is (the round model creeps toward
      // 83% and all four rounds land in the last seconds on one worker); the zone in view's cluster solved ahead
      // ends at its own units
      const units = unitsBound(s, this.computeN, now, a ? "view" : "all");
      if (units !== null) {
        const el = Math.max(1, now - s.sentAt);
        solveLeft = units.ms;
        solve = el / (el + units.ms);
        solveMs = el + units.ms;
        if (units.bounded && job.dataAt !== undefined) bounded = solve;
      }
      job.solveRounds = { first: st.first, rest: st.rest, measured: published >= 1 };
    }
    // T-1195: each stage's share of the bar is its expected time: the slot's last load's, else, before
    // the network is in, GUESS_SHARES, then the game stage as it took and the others as modelled. The
    // stage in hand moves by what has landed or its time against its expectation, never past its share.
    // (the model's data time scaled by how this browser's last load compared with it, T-1195)
    const dataMs = dlMs + openMs;
    // T-2574: a cached result expected to end the load: the game stage, the packs' download (their digests are the
    // cache's key) and the result's showing; nothing is opened or solved before the card goes
    const viaCache = job.viaCache !== undefined, gameExp = was?.game ?? GAME_MS * job.pace.game;
    if (viaCache && netIn && downloaded) job.cacheReadyAt ??= now;
    if (viaCache) {
      if (job.cacheNet !== netIn || !job.shares) job.shares = sharesOf({ game: netIn ? (job.netAt ?? job.since) - job.since : gameExp, data: dlMs, demand: CACHE_SHOW_MS });
      job.cacheNet = netIn;
    } else if (!job.shares || (!job.modelled && netIn)) {
      if (netIn) job.modelData = job.items.every(loaded) ? 0 : dataMs;
      const ms: StageMs | null = was ?? (netIn ? { game: (job.netAt ?? job.since) - job.since, data: job.modelData! * job.pace.data, demand: needsSolve ? prior : 0 } : null);
      job.shares = ms ? sharesOf(ms) : GUESS_SHARES;
      job.modelled = netIn;
    }
    // the data stage by its bytes and the packs open in every worker, or by its time against its expectation
    const openReal = openMs > 0 ? openDone / openMs : 1;
    const dataPart = !netIn && !job.items.length ? 0 : dataMs > 0 ? (dlMs * dl + openMs * openReal) / dataMs : 1;
    const shown = viaCache && this.cacheShown(), ready = job.cacheReadyAt;
    const parts: StageMs = {
      game: netIn ? 1 : within(job.since, gameExp),
      data: viaCache ? (downloaded ? 1 : Math.min(STEP_CAP, dl)) : job.dataAt !== undefined ? 1 : Math.min(STEP_CAP, Math.max(dataPart, netIn ? within(job.netAt, was?.data ?? dataMs * job.pace.data) : 0)),
      demand: viaCache ? (shown ? 1 : within(ready, CACHE_SHOW_MS)) : !needsSolve ? 1 : solve,
    };
    const shares = job.shares!;
    const fraction = Math.min(0.99, stagedFraction(shares, parts));
    barPlace(job, fraction, viaCache ? null : bounded);
    // T-1479: the time left, each stage's expected time less its time so far; a stage running late is
    // guessed to go on a share of its time so far. Unknown on a first load until its network is in.
    // T-1654: always a figure; a late round is estimated from the periods landed (`solveStage`).
    let target: number | null = null;
    if (viaCache) {
      const sinceGame = now - job.since, sinceReady = ready === undefined ? 0 : now - ready;
      target = (netIn ? 0 : Math.max(gameExp - sinceGame, 0.2 * sinceGame)) + (downloaded ? 0 : Math.max(300, dlMs - dlElapsed)) + (shown ? 0 : Math.max(CACHE_SHOW_MS - sinceReady, 0.2 * sinceReady));
    } else if (netIn || was) {
      const sinceGame = now - job.since;
      const gameLeft = netIn ? 0 : Math.max(gameExp - sinceGame, 0.2 * sinceGame);
      const dataExp = was?.data ?? dataMs * job.pace.data;
      const sinceData = job.netAt !== undefined ? now - job.netAt : 0;
      const dataLeft = job.dataAt !== undefined ? 0 : !netIn ? dataExp : Math.max(dataExp - sinceData, 0.3 * sinceData * (1 - dataPart));
      target = gameLeft + dataLeft + (needsSolve ? solveLeft : 0);
    }
    if (target !== null) job.left = { ms: nextLeft(job.left?.ms ?? null, target, job.left ? now - job.left.at : 0), at: now };
    const cityMb =job.items.filter((i) => !isMarket(i)).reduce((n, i) => n + (this.bytes.get(i) ?? 0), 0) / 1e6;
    this.barParts = { dlMs, dl, openMs, open, openReal, dataPart, solveMs, solve, prior, stops: stops ?? -1, lines: svc?.lines.length ?? -1, zones: sn?.size ?? -1, items: job.items.length, cityMb, workers: this.computeN, netIn: netIn ? 1 : 0, sinceNet: job.netAt !== undefined ? now - job.netAt : -1, sinceData: job.dataAt !== undefined ? now - job.dataAt : -1, elapsed: now - job.since, published: s && job.version === s.version ? s.published : -2, sent: s && job.version === s.version ? now - s.sentAt : -1,...Object.fromEntries(Object.entries(shares).map(([k, v]) => [`share_${k}`, v])), ...Object.fromEntries(Object.entries(parts).map(([k, v]) => [`part_${k}`, v])), was: was ? 1 : 0, viaCache: viaCache ? 1 : 0, fraction, bounded: bounded ?? -1, shown: job.best, leftTarget: target ?? -1, left: job.left?.ms ?? -1, modelSolve: job.modelSolve ?? -1, solveLeft }; // ?debug=1
    const cities = job.cities.filter((id) => !isMarket(id));
    noteLoad("demand " + stageInHand(parts), { zones: cities, workers: this.computeN }); // T-3927, T-3933
    loadProgress.value = { cities, fraction: job.best, step: STAGE_TEXT[stageInHand(parts)], startup: job.startup, leftMs: job.left?.ms ?? null };
  }

  // --- T-328: the demand cache (workers/demandCache.ts) -----------------------------------------

  /** The hash of a pack's bytes, waited for until it has downloaded. */
  private packHashOf(id: string) {
    let h = this.packHash.get(id);
    if (!h) {
      let set!: (s: string) => void;
      const p = new Promise<string>((r) => (set = r));
      this.packHash.set(id, (h = { p, set }));
    }
    return h;
  }

  private hashPack(id: string, parts: Uint8Array[]) {
    if (!this.cacheOn) return;
    const h = this.packHashOf(id);
    Promise.all(parts.map((b) => sha256(b as Uint8Array<ArrayBuffer>))).then((hs) => h.set(hs.join("")), () => h.set("unhashed"));
  }

  /** The cache key of a solve covering `cities` (and those zones' market packs), once their packs
   * have downloaded. T-1223: a zone solved coarse (`coarse`) keys its pack's hash with ":coarse". */
  private async keyFor(cities: CityId[], markets: CityId[], coarse: CityId[]): Promise<CacheKey> {
    const ids = [...markets.map(marketItem), ...cities];
    const hashes = await Promise.all(ids.map((id) => this.packHashOf(id).p));
    const c = new Set<string>(coarse);
    return { ...this.keyBase(), packs: Object.fromEntries(ids.map((id, i) => [id, c.has(id) ? `${hashes[i]}:coarse` : hashes[i]])) };
  }
  private keyBase(): Omit<CacheKey, "packs"> {
    return { format: CACHE_FORMAT, demand: DEMAND_KEY, workers: this.poolN, flags: new URLSearchParams(location.search).get("perfOff") ?? "" };
  }

  /** T-2574: a game load's cached result is expected (`log` found it, its packs still to check): the card's bar
   * counts to its showing (`jobProgress`), the card going with it (`cacheDone`). Not with the old card. `on` false:
   * that read (any, for null) is no longer expected to show. */
  private expectCache(log: object | null, on: boolean) {
    const job = this.job;
    if (!job || job.wait === null) return;
    if (on) { if (!perfOff("cacheCard")) job.viaCache = log!; }
    else if (job.viaCache && (!log || job.viaCache === log)) {
      // (the full load's shares again, from where the bar stands)
      job.viaCache = job.cacheNet = job.cacheReadyAt = undefined;
      job.modelled = false;
    }
  }

  /** A whole game's network arrived: show its cached final result if there is one for exactly this
   * network, these packs and this build, and nothing of its solve is shown yet. */
  private async tryCache(svc: Service, ahead?: StoredDemand) {
    const log: NonNullable<DemandClient["cacheLog"]> = (this.cacheLog = { at: performance.now(), ...(ahead ? { ahead: true } : {}) });
    this.cacheLogs.push(log);
    // T-1539: found ahead of the network, it stands while that save still loads and no network came
    const late = () => (ahead ? this.early?.svc !== svc || !gameLoadArmed() : this.latest() !== svc || (this.cur?.svc === svc && this.cur.published >= 0 && !this.cur.rough));
    try {
      const canon = canonical(svc);
      const hash = (log.hash = await sha256(canon.text));
      const stored = ahead ?? await readDemandCache<StoredDemand>(hash, saveSlotsState.peek().active);
      log.readMs = performance.now() - log.at;
      if (!stored) return void (log.miss = "none");
      // the zones it covered must be the ones this solve will cover
      await loadFootprints(this.nearOf(svc).near);
      const need = this.solveNeed(svc);
      // T-2588: ahead of the network, a zone opened before the reach was known that this network neither
      // reaches nor views is not counted: closeStrays closes it before the load's solve
      if (ahead && need) {
        const reached = this.reachedSet(svc), viewed = this.viewedSet();
        if (reached) for (const id of this.unknownReach) if (!reached.has(id) && !viewed.has(id)) need.delete(id);
      }
      const cities = Object.keys(stored.packs ?? {}).filter((k) => !isMarket(k));
      if (!need || need.size !== cities.length || cities.some((id) => !need.has(id))) return void (log.miss = "zones");
      const markets = this.wantsMarket(svc);
      const had = Object.keys(stored.packs ?? {}).filter(isMarket).map(marketZone);
      if (had.length !== markets.length || markets.some((id) => !had.includes(id))) return void (log.miss = "zones");
      // (T-2574: what can be checked before the packs are in)
      const pre = sameKey(stored, { ...this.keyBase(), packs: stored.packs ?? {} });
      // T-4202: after an update, the last build's result of this network stands in while this one solves it
      const standIn = (pre === "demand" || pre === "workers") && !perfOff("standIn");
      if (pre && !standIn) return void (log.miss = pre);
      if (!late()) this.expectCache(log, true);
      // (T-1223: the levels the solve will have; before the fine set is made, the save's list's)
      const saved = !this.lodZones && this.lod === "auto" ? this.savedFine : null;
      const coarse = cities.filter((id) => (saved ? !saved.includes(id) : this.levelFor(id) === LOD_COARSE));
      const key = await this.keyFor(cities, markets, coarse);
      const why = !standIn ? sameKey(stored, key) : standsIn(stored, key) ? null : sameKey({ ...stored, demand: key.demand, workers: key.workers }, key);
      if (why) return void (log.miss = why);
      if (standIn) log.standIn = pre!;
      // (T-1533: a rough solve's estimate gives way to it)
      if (late() || this.failed) return void (log.miss = "late");
      this.applyCache(svc, canon, stored, hash, cities, standIn);
      if (ahead && this.early) this.early.shown = true;
      log.shownAt = performance.now();
    } catch (err) {
      log.miss = "error: " + String(err);
    } finally {
      if (log.miss) this.expectCache(log, false);
    }
  }

  /** Show a cached result as a final one under CACHE_VERSION; the solve of the same network carries
   * on, its estimates are not shown, and its final round replaces it whole (T-4202: a stand-in's too). */
  private applyCache(svc: Service, ids: ReturnType<typeof canonical>, stored: StoredDemand, hash: string, cities: CityId[], standIn = false) {
    const days = fromStored(stored, ids);
    if (days.length !== DEMAND_DAYS.length) throw new Error("days");
    const now = performance.now();
    // a solve entry for it, so its layout and zones are known as a shown solve's are (T-324, T-325);
    // first in the log, so the newest solve stays last
    this.log.unshift({ version: CACHE_VERSION, svc, layout: packService(svc).layout, results: [], byRound: [], published: stored.round, arrivedAt: now, sentAt: now, periodMs: [], setupMs: [], memBytes: [], cities, markets: Object.keys(stored.packs ?? {}).filter(isMarket).map(marketZone), showEstimates: false });
    this.cachedCities = new Set(cities);
    this.cachedHash = hash;
    this.cachedModes = stored.modes ?? null;
    seedRows(stored.rows ?? []);
    this.cachedAhead = stored.ahead?.save ?? null;
    this.cachedStandIn = standIn;
    this.cachedLocal = !!stored.local;
    if (stored.local) { this.localEdits = 1; this.editAt = performance.now(); this.refreshSoon(); }
    if (this.cur?.svc === svc) this.cur.showEstimates = false;
    // T-2045: every zone has its final figures; T-2533: the focus pool stays for the zone in view's questions
    this.hideFocus();
    // T-1533: a rough solve under way gives way, so the packs it holds back open at once (a rough
    // solve of no zone, without the barrier, ends the worker's at its next look)
    if (this.cur?.rough) {
      const version = ++this.version;
      for (const w of this.workers) w.postMessage({ kind: "solve", version, ...(({ net, ids }) => ({ net, ids }))(packService(this.cur.svc)), rounds: ROUNDS, only: [] } satisfies ToDemand);
    }
    this.days = days;
    this.lodShown(cities, cities.filter((id) => this.levelFor(id) === LOD_COARSE));
    this.addBareViews();
    demandUpdating.value = true;
    // T-2560: the card and the held clock go now, once the network is in (`cacheDone`)
    this.progress();
    this.checkHold();
  }

  /** The figures shown are the cache's. */
  private cacheShown(): boolean {
    return this.days[0]?.view?.version === CACHE_VERSION;
  }

  /** A final result landed: keep it beside the save (not from a test page or a tab without the slot). */
  private async writeCache(s: Solve) {
    const slots = saveSlotsState.peek();
    const skip = (why: string) => void ((this.cacheLog ??= { at: performance.now() }).skip = why);
    if (!this.cacheOn || !this.cacheWrites || !slots.writable || !slots.active) return skip("off");
    const days = this.days;
    if (days.some((d) => d.view?.version !== s.version)) return skip("days");
    // (the last asked writes: one asked before it can finish its reads after it, T-4266)
    const seq = ++this.cacheSeq;
    // (T-4260: held local whether or not this solve ended local: another city's whole solve leaves the figures of one an
    // edit left local as they are)
    const local = !!s.localHeld?.slice(0, this.computeN).some(Boolean);
    const t0 = performance.now();
    const canon = canonical(s.svc), sent = s.sent ?? s;
    const stored = toStored(days, sent.cities, canon);
    const [hash, key] = await Promise.all([sha256(canon.text), this.keyFor(sent.cities, sent.markets, s.coarse ?? [])]);
    // the bubbles, when they were asked for this result (the day shown, every purpose, the viewed zone)
    const m = this.modesP, city = this.focusCity;
    const cm = m && m.version === s.version ? await m.p : null;
    // T-1539: the network the loaded save worked out, so the next load of it shows this at once
    const lk = this.loadKey;
    const ahead = lk?.snapshot && lk.svcKey === keyOf(s.svc) ? { save: lk.key, snapshot: lk.snapshot } : null;
    if (seq !== this.cacheSeq) return skip("newer");
    // the result shown at load is the same one, unless it can now add its bubbles or its network, or it stood in (T-4202)
    if (hash === this.cachedHash && !this.cachedStandIn && this.cachedLocal === local && (this.cachedModes || !cm) && (!ahead || this.cachedAhead === ahead.save)) return skip("same");
    const modes = cm && { city, data: concat([cm.homeRail, cm.homeWalk, cm.homeDrive, cm.workRail, cm.workWalk, cm.workDrive, cm.homeNone, cm.workNone]), ...(m!.day ? { day: m!.day } : {}) };
    const ms = performance.now() - t0;
    const bytes = storedSize(stored) + (modes?.data.byteLength ?? 0);
    // T-2136: with the zone in view's and the built zones' corridor rows, which no worker has at the next load until it holds the market
    const bookKey = days[0].intercity?.corridors.book.key;
    const rows = bookKey ? heldRows(bookKey, cityById(activeCity.peek())?.place ?? -1, CACHED_ROWS) : [];
    const ok = await this.writeStored(slots.active, hash, { ...key, at: Date.now(), round: s.published, days: stored, ...(modes ? { modes } : {}), ...(ahead ? { ahead } : {}), ...(rows.length ? { rows } : {}), ...(local ? { local: true as const } : {}) } satisfies StoredDemand);
    this.cachedHash = hash;
    this.cachedLocal = local;
    this.cachedAhead = ahead?.save ?? null;
    this.cachedStandIn = false;
    (this.cacheLog ??= { at: t0 }).wrote = { at: performance.now(), ms, bytes, ok, ...(ahead ? { ahead: true } : {}) };
  }

  /**
   * T-789: the cache written by a worker of its own (`cacheWrite.worker.ts`), the corridors copied a
   * day per task and transferred, since cloning them into IndexedDB took the main thread longer
   * than a frame. Where no worker starts, written here as before.
   */
  private async writeStored(slot: string, hash: string, v: StoredDemand): Promise<boolean> {
    if (!this.cacheWriter) {
      try {
        const w = new Worker(new URL("./cacheWrite.worker.ts", import.meta.url), { type: "module" });
        w.onmessage = (e: MessageEvent<{ id: number; ok: boolean }>) => { this.cacheWriteWaits.get(e.data.id)?.(e.data.ok); this.cacheWriteWaits.delete(e.data.id); };
        w.onerror = () => { for (const done of this.cacheWriteWaits.values()) done(false); this.cacheWriteWaits.clear(); };
        this.cacheWriter = w;
      } catch {
        return writeDemandCache(slot, hash, v);
      }
    }
    const transfer: ArrayBuffer[] = [];
    const days: StoredDemand["days"] = [];
    // T-1008: the days share the market's book, copied once (the clone keeps it shared)
    const books = new Map<CorridorBook, CorridorBook>();
    const own = <T extends Uint16Array | Uint32Array | Float64Array | Float32Array>(a: T): T => { const b = a.slice() as T; transfer.push(b.buffer as ArrayBuffer); return b; };
    for (const d of v.days) {
      const c = d.intercity?.corridors;
      if (c) {
        let t = books.get(c.book);
        if (!t) books.set(c.book, (t = { key: c.book.key, n: c.book.n, totals: own(c.book.totals), own: own(c.book.own), apart: own(c.book.apart) }));
        const copy: Corridors = { book: t, longerScale: c.longerScale, commuteScale: c.commuteScale, railAt: own(c.railAt), longerRail: own(c.longerRail), commuteRail: own(c.commuteRail), railFrom: own(c.railFrom), railTo: own(c.railTo), railAir: own(c.railAir) };
        days.push({ ...d, intercity: { ...d.intercity!, corridors: copy } });
      } else days.push(d);
    }
    if (v.modes) transfer.push(v.modes.data.buffer as ArrayBuffer);
    const rows = v.rows?.map(({ place, rows: r }) => ({ place, rows: { key: r.key, n: r.n, at: own(r.at), from: own(r.from), to: own(r.to), longer: own(r.longer), commute: own(r.commute), air: own(r.air), car: own(r.car) } }));
    const id = ++this.cacheWriteId;
    return new Promise((resolve) => {
      this.cacheWriteWaits.set(id, resolve);
      this.cacheWriter!.postMessage({ id, slot, hash, value: { ...v, days, ...(rows ? { rows } : {}) } }, transfer);
    });
  }

  /**
   * A held game's figures are final (T-347): its network's solve has published HOLD_ROUND and the
   * viewed zone is open, so its figures show. A zone the solve covers has them from that round; a
   * zone open bare (T-325) from its opening, since they are final as they come (no station reaches
   * it: what any solve gives it). T-351: a bare start's solve covers no zone, only the market, and
   * lands within a second; its hold ends when the viewed zone's bare figures arrive.
   */
  private holdDone(): boolean {
    const s = this.cur;
    if (!this.hold || !s || this.hold.version !== s.version || this.waiting) return false;
    return this.viewHasFinal(s) && [...this.viewedSet()].every((id) => this.isOpen(id));
  }

  /** T-421: the zone in view has its final: the solve's, or its cluster's solved ahead (T-402). */
  private viewHasFinal(s: Solve): boolean {
    if (s.rough) return false;
    if (s.published >= HOLD_ROUND) return true;
    const a = s.ahead, c = a && s.clusterRes?.get(a.key);
    // T-2459: with the city view's zone's cluster ahead of the zone in view's, the zone in view's
    // focus figures settled
    if (this.selectedAhead(s) && [...this.viewedSet()].every((id) => focusZones.peek().get(id) && this.fshown.get(id)?.s.svc === s.svc)) return true;
    return !!c && c.published >= ROUNDS && [...this.viewedSet()].every((id) => a.cities.includes(id));
  }

  /** T-2459: `s` solves the city view's zone's cluster ahead of the zone in view's */
  private selectedAhead(s: Solve): boolean {
    const a = s.ahead, v = [...this.viewedSet()];
    return !!s.focused && !!a && v.length > 0 && v.some((id) => !a.cities.includes(id));
  }

  /**
   * T-2560: a held game shows the cached result of its own network (T-328) and the zone in view has
   * figures: that network, whose hold may go now. The cache is the final of exactly this network, build
   * and packs, so the fresh final equals it; T-347 held the card through that solve for its lag.
   */
  private cacheDone(): Service | null {
    if (!this.hold || this.failed || !this.cacheShown() || perfOff("cacheCard")) return null;
    const c = this.log.find((s) => s.version === CACHE_VERSION)?.svc;
    if (!c || this.latest() !== c) return null;
    const shown = cityDemandViews.peek();
    return [...this.viewedSet()].every((id) => this.cachedCities.has(id) || (this.isOpen(id) && shown.has(id))) ? c : null;
  }

  /** A held game's network has its final result (T-347), or its zone in view has (T-421), or it shows
   * its network's cached result (T-2560): the clock runs. */
  private checkHold() {
    if (this.hold && this.idle()) return this.settleIdle(); // T-318
    const done = this.holdDone(), cached = !done ? this.cacheDone() : null;
    if (done || cached) {
      this.hold = null;
      const s = this.cur;
      if (cached) {
        // the load ends with the card, unremembered: its time says nothing of a load without the cache
        this.cacheSolve = cached;
        this.endJob();
        if (clock?.held.peek()) endCardFull(loadProgress.peek()); // (T-2574)
        loadProgress.value = null;
        this.zoneProgress();
      } else if (s && s.published < ROUNDS) {
        const solving = demandSolving.peek();
        this.unsettled = new Set(s.cities.filter((id) => solving?.has(id) ?? true));
        // their bars start from their own solve's share (the card's load ended without them)
        for (const id of this.unsettled) this.zonePlaces.delete(id);
        // the load in hand ends with the card: the other zones show their own bars (`zoneProgress`)
        this.endJob();
        loadProgress.value = null;
        this.zoneProgress();
      }
      loadComplete();
      // the nearby zones' packs may load now (T-276), and those track crosses (T-1047)
      if (nearbyZones.peek().length || trackZones.peek().built.size) this.ensure();
      this.prefetchViewed(); // a building tool picked while the card was up (T-351)
      // T-1223: levels wait for a load's card; T-1733: and the load's own time is no zone's time left
      // alone (all of Japan's 17 minutes made 44 of 45 zones coarse the moment it ended)
      const now = Date.now();
      for (const id of this.fine) if (this.touched.has(id)) this.touched.set(id, now);
      this.lodSoon();
    }
  }

  /**
   * A change of focus ends the old city's cached answers and rejects its late replies. Checked by
   * every query as well as by the effect on `activeCity`: effects run in no useful order, so a view
   * that reacts to the same change could otherwise read the old city's cached cells (T-118: the
   * commuter bubbles showed New York's cells for a Philadelphia selection).
   */
  private syncFocus() {
    const c = activeCity.peek();
    const frame = worldFrameEpoch.peek();
    if (c === this.focusCity && frame === this.focusFrame) return;
    this.focusFrame = frame;
    // T-1533: the zone now in view, if its pack waits to open in a worker, opens next there
    if (c !== this.focusCity && ((this.fetching.has(c) && !this.opened.has(c)) || (this.fetchingBare.has(c) && !this.bareOpen.has(c)))) for (const w of this.workers) w.postMessage({ kind: "first", id: c } satisfies ToDemand);
    this.focusCity = c;
    this.focusEpoch++;
    this.wakeFocusAsks();
    this.cellsP = null;
    this.modesP = null;
    this.purposeP.clear();
  }

  /** T-2453: a zone newly shown in the city view, if its pack waits to open in a worker, opens next
   * there after the zone in view's (the workers keep earlier `first` packs ahead of it) */
  private selectedFirst() {
    const c = this.selectedZone();
    if (c === this.selFirst) return;
    this.selFirst = c;
    if (c && ((this.fetching.has(c) && !this.opened.has(c)) || (this.fetchingBare.has(c) && !this.bareOpen.has(c)))) for (const w of this.workers) w.postMessage({ kind: "first", id: c } satisfies ToDemand);
  }
  private selFirst: CityId | null = null;

  /** Stop the pool; also called for a load that cannot finish for another reason (T-950). */
  fail(why: string) {
    if (!this.status.startsWith("failed")) this.statusBefore = this.status;
    this.status = "failed: " + why;
    demandUpdating.value = false;

    this.onOpen.splice(0).forEach(w => w.f(false));
    for (const q of this.pendingQueries.values()) q.done([]);
    this.pendingQueries.clear();
    this.releaseRespawnWaits();
    this.endJob();
    this.hold = null;
    this.cacheSolve = null;
    this.endFocus();
    this.endAside();
    this.dropSide();
    this.shownSnap.clear(this.version); // (T-4220)
    this.pushed.clear();
    zoneLoads.value = new Map();
    console.error("demand:", why);
    loadFailed(why);
  }

  // --- T-1174: a worker whose WASM panicked is replaced ---------------------------------------------

  /** The workers being replaced: `left`, the items a replacement has still to reopen (null before
   * they are sent); `solving`, reopened and sent the solve again; `planned`, restarted to give back
   * memory (T-1205), not after a panic. Each is cleared by its replacement's first `done`. */
  private respawn = new Map<number, { left: Set<string> | null; solving: boolean; planned: boolean; held?: boolean }>();
  /** a replacement whose solve waits for a game load's network (`held`) is no sign that a fresh worker fails */
  private respawnTrusted(r: { held?: boolean }) { return !!r.held && gameLoadArmed(); }
  private respawns = 0;
  /** T-3492: questions waiting for the solve that follows a replacement to be sent (`reopening`) */
  private respawnWaits: (() => void)[] = [];
  /** a replacement is still opening what its worker held: the pool can answer nothing yet */
  private reopening(): boolean {
    return !this.failed && [...this.respawn.values()].some((r) => !r.solving);
  }
  private releaseRespawnWaits() {
    if (this.reopening()) return;
    for (const r of this.respawnWaits.splice(0)) r();
  }
  /** the next solve has every worker make its market parts afresh (`reopened`, `shrink`) */
  private freshParts = false;

  /** Worker `i`'s WASM panicked (or ran out of memory), leaving its objects unusable: a new worker
   * takes its place, reopens what it held and the solve is sent again. A second panic before a
   * replacement has seen a solve through, or more than MAX_RESPAWNS, stops the pool as before: the
   * cause is then one a fresh worker does not cure (most likely memory, T-1169). `planned`
   * (T-1205): an idle worker restarted to give back its memory, which counts toward neither. */
  private workerDied(i: number, why: string, planned = false, download = false) {
    if (this.failed) return;
    demandNote("worker-restart", { worker: i, planned, download, why }); // T-4206
    if (!planned) this.lastTrouble = `worker ${i}${download ? " download" : ""}: ${why}`; // i18n-ignore
    if (!planned && !download && this.computeN > 1 && isMemoryFailure(why)) return this.memoryShort(i, why);
    const panicked = [...this.respawn.values()].some((r) => !r.planned && !this.respawnTrusted(r));
    if (!planned && !download && (panicked || this.respawns >= MAX_RESPAWNS)) return this.fail(`worker ${i}: ${why}`); // i18n-ignore
    if (download) console.warn(`demand: worker ${i}'s code did not download; trying again`); // i18n-ignore
    else if (planned) { console.info(`demand: worker ${i} restarted to give back memory: ${why}`); memoryResolve.value = true; }
    else { console.warn(`demand: worker ${i} panicked, replaced: ${why}`); this.respawns++; }
    // (a download's replacement counts as planned: it neither stops the pool at a panic nor counts as one)
    this.respawn.set(i, { left: null, solving: false, planned: planned || download });
    this.workers[i].terminate();
    if (this.isReady.delete(i)) this.ready--;
    this.endQueries();
    this.workers[i] = this.spawn(i, false);
  }

  /** T-2867: worker `i` could not get memory, however much the machine has (a 32-bit browser's address space, a
   * full page file): half the solving workers go, the focus pool with them, and the rest solve again;
   * no figure changes. Workers being replaced start again at their place in the smaller pool. With one
   * solving worker left, `workerDied` treats it as any panic, and a second stops the pool. */
  private memoryShort(i: number, why: string) {
    const aux = this.computeN, n = Math.max(1, Math.floor(aux / 2));
    console.warn(`demand: worker ${i} ran out of memory, ${aux} solving workers to ${n}: ${why}`); // i18n-ignore
    this.endFocus();
    this.endAside();
    rememberPoolCap(n);
    const again = new Set([...this.respawn.keys(), i]);
    this.respawn.clear();
    this.shrink(n, why);
    for (const k of again) {
      if (k >= n && k !== aux) continue; // (stopped with the shrink)
      const j = k === aux ? n : k;
      this.respawn.set(j, { left: null, solving: false, planned: false });
      this.workers[j].terminate();
      if (this.isReady.delete(j)) this.ready--;
      this.workers[j] = this.spawn(j, false);
    }
    if (!this.respawn.size) this.solveAgain();
  }

  /** T-2704: worker `i`'s code did not download (a dropped fetch): a new worker takes its place after a
   * wait, as `workerDied` replaces one, reopening what it should hold (anything posted to it was lost);
   * after SCRIPT_WAITS_MS's tries for that worker the pool stops. */
  private scriptFailed(i: number) {
    if (this.failed) return;
    const k = this.scriptTries[i] ?? 0;
    if (k >= SCRIPT_WAITS_MS.length) return this.fail(`worker ${i}: its code could not be downloaded`); // i18n-ignore
    this.scriptTries[i] = k + 1;
    const dead = this.workers[i];
    setTimeout(() => { if (this.workers[i] === dead) this.workerDied(i, "its code did not download", false, true); }, SCRIPT_WAITS_MS[k]); // i18n-ignore
  }
  private scriptTries: number[] = [];

  /** Questions waiting on a worker's answer end empty, as at a failure; the views ask again. */
  private endQueries() {
    for (const q of this.pendingQueries.values()) q.done([]);
    this.pendingQueries.clear();
    this.modesP = null;
    this.purposeP.clear();
    this.aheadP.clear();
  }

  private reopenBusy = 0;
  private reopenQueue: (() => void)[] = [];
  /** Run one of a replacement's downloads when one of REOPEN_FETCHES places is free; `stale` (the replacement
   * itself was replaced, or the pool stopped) ends the wait with an error the caller ignores. */
  private async reopenSlot<T>(stale: () => boolean, run: () => Promise<T>): Promise<T> {
    if (this.reopenBusy >= REOPEN_FETCHES) await new Promise<void>((go) => this.reopenQueue.push(go)); // (the place is handed on, not freed)
    else this.reopenBusy++;
    try {
      if (stale()) throw new Error("replaced");
      return await run();
    } finally {
      const next = this.reopenQueue.shift();
      if (next) next(); else this.reopenBusy--;
    }
  }

  /** The replacement for worker `i` is ready: download again what the worker held (the page keeps no
   * copy; the browser's cache has them) and open it there. */
  private async reopen(i: number) {
    const r = this.respawn.get(i)!, aux = i === this.computeN;
    const items = [...this.fetching].filter((x) => (!aux || !isMarket(x)) && this.holds(i, x));
    const bare = i === this.auxAt ? [...this.fetchingBare] : [];
    // the downloads of every replacement share REOPEN_FETCHES places (T-3604: all of them at once, in three
    // workers, ran the browser out of connections: ERR_INSUFFICIENT_RESOURCES, and the replacements never opened)
    const stale = () => this.respawn.get(i) !== r || this.failed;
    const city = (id: CityId) => this.reopenSlot(stale, async () => {
      const [header, bin] = await this.downloadPack(id);
      return { id, header: new TextDecoder().decode(header), bin: bin.slice().buffer as ArrayBuffer, level: this.levelPosted.get(id) ?? LOD_FINE };
    });
    try {
      const markets = items.filter(isMarket).map(marketZone);
      const [packs, barePacks, intercity, rows, drives] = await Promise.all([
        Promise.all(items.filter((x) => !isMarket(x)).map((id) => city(id as CityId))),
        Promise.all(bare.map(city)),
        Promise.all(markets.map((id) => this.reopenSlot(stale, async () => ({ id, pack: await this.download(`${id}.market.json`) })))),
        markets.length ? this.download(MARKET_ROWS) : Promise.resolve(undefined),
        markets.length ? fetchMarketDrives().then((d) => d ?? undefined) : Promise.resolve(undefined),
      ]);
      if (this.respawn.get(i) !== r || this.failed) return;
      // closed meanwhile: left out
      const full = packs.filter((p) => this.fetching.has(p.id)), mk = intercity.filter((m) => this.fetching.has(marketItem(m.id)));
      const bares = barePacks.filter((p) => this.fetchingBare.has(p.id));
      r.left = new Set([...full.map((p) => p.id as string), ...mk.map((m) => marketItem(m.id)), ...bares.map((p) => p.id as string)]);
      const w = this.workers[i];
      const corridors = mk.length && this.corridorFile ? this.corridorFile.slice() : undefined;
      if (full.length || mk.length) w.postMessage({ kind: "open", packs: full, ...(mk.length ? { intercity: mk, rows, ...(drives ? { drives } : {}), ...(corridors ? { corridors } : {}) } : {}) } satisfies ToDemand, [...full.map((p) => p.bin), ...mk.map((m) => m.pack.buffer as ArrayBuffer), ...(mk.length && rows ? [rows.buffer as ArrayBuffer] : []), ...(mk.length && drives ? [drives.buffer as ArrayBuffer] : []), ...(corridors ? [corridors.buffer as ArrayBuffer] : [])]);
      if (bares.length) w.postMessage({ kind: "open", packs: bares, bare: true } satisfies ToDemand, bares.map((p) => p.bin));
      if (!r.left.size) this.reopened(i);
    } catch (err) {
      if (this.respawn.get(i) === r) this.fail(`worker ${i}: its replacement could not reopen its packs: ${String(err)}`); // i18n-ignore
    }
  }

  /** The replacement holds what worker `i` held: once every replacement does, the pool is whole again
   * and the newest network is solved again under a new version (the shown figures stay until its final). */
  private reopened(i: number) {
    const r = this.respawn.get(i)!;
    r.left = null;
    r.solving = true;
    if (!this.isReady.has(i)) { this.isReady.add(i); this.ready++; }
    this.solveAgain();
  }

  /** The pool changed under a solve (`reopened`, `memoryShort`): once every worker is ready, the newest
   * network is solved again. */
  private solveAgain() {
    if (this.ready < this.workers.length) return;
    this.status = "ready";
    this.settleWaiters();
    // the replacement composed its market once from every pack, where the others may still hold
    // parts of older compositions (their corridor tables then differ): all make theirs afresh
    this.freshParts = true;
    if (this.waiting) this.flush(true);
    // (T-4199: a game load's network on its way is solved when it comes, as `flush` sends nothing until then)
    else if (gameLoadArmed()) { for (const r of this.respawn.values()) r.held = true; this.releaseRespawnWaits(); }
    // (T-3373 B: a shrink moves packs, which open again before the solve can cover them)
    else if (this.cur && this.holding) { this.waiting = { svc: this.cur.svc, arrivedAt: performance.now() }; this.flush(true); }
    else if (this.cur) this.dispatch(this.cur.svc, performance.now());
    else { this.respawn.clear(); this.releaseRespawnWaits(); }
  }

  /** `item` is open in no worker. */
  private unack(item: string) {
    this.acks.delete(item);
    for (const by of this.ackedBy) by?.delete(item);
  }

  /** T-3781: the loading card's watchdog found nothing moving. Once per page, every worker is replaced (as a
   * planned restart: not counted as a panic), reopens what it held and the newest network is solved again; the
   * focus pool stops (its figures stay shown). False when there is
   * nothing to restart, or it was done already: the failure screen then goes up. */
  restartStuck(): boolean {
    if (this.stuckRestarted || this.failed || !this.workers.length || [...this.respawn.values()].some((r) => !this.respawnTrusted(r))) return false;
    this.stuckRestarted = performance.now();
    const quiet = this.diagnosis();
    this.lastTrouble = `stuck, workers restarted: ${quiet}`.slice(0, 600); // i18n-ignore
    console.warn("demand: nothing moved while loading, restarting the workers:", quiet); // i18n-ignore
    this.endFocus();
    this.endAside();
    this.endQueries();
    for (let i = 0; i < this.workers.length; i++) {
      this.respawn.set(i, { left: null, solving: false, planned: true });
      this.workers[i].terminate();
      if (this.isReady.delete(i)) this.ready--;
      this.heard[i] = undefined;
      this.inUnit[i] = null;
      this.workers[i] = this.spawn(i, false);
    }
    return true;
  }

  /** T-3869: a save found no memory. Every worker is replaced at once (`restartStuck`'s path: the focus pool stops,
   * the replacements reopen what the workers held and the newest network is solved again; the figures shown stay),
   * which gives back all they held while the replacements start, the moment the save is tried again. Half the
   * solving workers go for the tab's session, as when a worker itself runs out (T-2867). */
  restartForSave() {
    if (this.failed || !this.workers.length) return;
    console.warn("demand: a save found no memory; the workers are replaced"); // i18n-ignore
    this.endFocus();
    this.endAside();
    this.endQueries();
    if (this.computeN > 1 && !this.respawn.size) {
      const n = Math.floor(this.computeN / 2);
      rememberPoolCap(n);
      this.shrink(n, "a save found no memory"); // i18n-ignore
    }
    for (let i = 0; i < this.workers.length; i++) {
      this.respawn.set(i, { left: null, solving: false, planned: true });
      this.workers[i].terminate();
      if (this.isReady.delete(i)) this.ready--;
      this.heard[i] = undefined;
      this.inUnit[i] = null;
      this.workers[i] = this.spawn(i, false);
    }
    if (demandView.peek()?.final) memoryResolve.value = true;
  }

  // --- T-4220: an edit made during a whole solve, solved aside against the figures on screen ----------

  /**
   * T-4220 (notes/T-4220.md, "The pick" 5): the side job's pool, one worker holding up to `ASIDE_MAX_ZONES` zones. Never
   * alive beside `fpool` (T-2045's and T-2605's): it ends a refine aside as it starts; a refine wanted while no side job
   * is in hand or about to be decided ends it in turn (`refineAside`, T-4307).
   * An edit does not end it: its zones and its copy of the snapshot stay for the next edit made during the same whole
   * solve. A side job ending, or a final shown, with no whole solve running or no zone held there ends it (`asideIdle`),
   * as does the main pool showing the last side job's network or a newer one, or that of the side job in hand, whose
   * figures could then never be shown (`sideCovered`). It answers no question: a side job leaves its zones' populations
   * there on its own network (T-4297).
   */
  private apool: FocusPool | null = null;
  /** the level each zone was posted to it at, and those opening there */
  private alevel = new Map<CityId, number>();
  private aopening = new Map<CityId, () => void>();
  /** the exchange in hand with it: each of its messages goes here until it answers (null: the pool ended) */
  private atake: ((msg: FromDemand | null) => void) | null = null;
  /** the side job in hand, or a dev call's: one at a time */
  private asideJob: { svc: Service; gen: number } | null = null;
  /** the snapshot generation each network's side job was decided on (started or refused): once a generation */
  private asideDecided = new WeakMap<Service, number>();
  /** the pool failed: none starts again until a whole snapshot comes after; `ASIDE_FAILS` failures in a game, none until
   * another game (a worker that runs out of memory on its zones would otherwise start, and fail, at every whole solve) */
  private asideBroken = false;
  private asideFails = 0;
  /** T-4307: networks whose side job was decided and shows nothing (`asideNothing`): T-2605's refine of an edit may run */
  private asideNone = new WeakSet<Service>();
  /** a side job waits for the running solve's time left (`unitsLeft`), looked at again at its next `work` */
  private asideWaitsLeft = false;
  private asideTimer = 0;
  /** ?debug=1: each side job decided, oldest first */
  asideLog: AsideLog[] = [];

  /** T-4220: the network a side job is wanted for and the running whole solve it would beat: (a) a network held behind
   * that solve (`firm`; not one taken to go at its local result, `quickHold`), or (b) that solve's own network when it
   * shows no local result (started again after T-3925's cut, or an edit solved whole at once): every solving worker's
   * plan is in and one has no local stage, which sends them all to the whole solve at once (T-1733; stage 8: with
   * several workers one whose populations outlived the cut, or with a smaller share of the edit, can still have one) */
  private asideWanted(): { svc: Service; s: Solve; held: boolean } | null {
    const s = this.cur;
    if (!s || s.barrier !== false || s.rough || s.published >= ROUNDS || this.hold || gameLoadArmed()) return null;
    const w = this.waiting;
    if (w) return w.svc !== s.svc && this.firm() && !this.quickHold(s) ? { svc: w.svc, s, held: true } : null;
    if (s.refresh) return null;
    const plans = Array.from({ length: this.computeN }, (_, k) => s.localPlans?.[k]);
    return plans.every((v) => v !== undefined) && plans.some((v) => v === 0) ? { svc: s.svc, s, held: false } : null;
  }

  /** T-4220: why no side job may run here now, or null */
  private asideOff(): string | null {
    if (perfOff("editAside") || perfOff("focus")) return "off";
    if (smallDevice() || addressLimited()) return "device";
    // (`?split=1`: T-3373 B's solves are all whole, round by round, with no snapshot)
    if (this.holding) return "split";
    if (this.failed || this.asideBroken || this.asideFails >= ASIDE_FAILS) return "pool";
    return null;
  }

  /** T-4220: look for a side job once the edits have been quiet `ASIDE_QUIET_MS` */
  private asideSoon() {
    clearTimeout(this.asideTimer);
    this.asideTimer = 0;
    if (this.asideJob || !this.asideWanted()) return;
    const wait = this.editAt + ASIDE_QUIET_MS - performance.now();
    if (wait > 0) this.asideTimer = window.setTimeout(() => this.asideSoon(), wait + 5);
    else this.asideGo();
  }

  /** T-4220: the zones the main pool's solve of `svc` would cover, in the registry's order: those open and those it
   * needs (one not open yet is refused by the plan, as one opened since the snapshot); null while its reach is not known */
  private asideCities(svc: Service): CityId[] | null {
    const need = this.solveNeed(svc);
    return need && CITIES.map((c) => c.id).filter((id) => this.opened.has(id) || need.has(id));
  }

  /**
   * T-4220: a side job for the network wanted now, decided once per network and snapshot generation: once the running
   * solve's snapshot is in from every solving worker and whole on the page, and that solve's time left is known. Refused
   * on the page (`asideOff`, the fares changed since the figures on screen, the side pool's memory, too little left to
   * gain), else planned in the side pool; solved there if the plan has fresh zones held at the main pool's levels now,
   * they fit in memory too, and the side job's expected time is under `ASIDE_GAIN` of the main pool's (`asideRun`).
   * With no side job in hand after it, T-2605's refine may run (`refineAside`).
   */
  private asideGo() {
    try { this.asideDecide(); } finally { if (!this.asideJob) this.refineAside(); }
  }
  private asideDecide() {
    const want = this.asideWanted();
    if (!want || this.asideJob) return;
    const { svc, s, held } = want;
    // (`shownIn` looks again once the running solve's snapshot is in, the next `work` once its time left is known; a
    // solve whose workers were not asked for it has none coming)
    if (this.shownSnap.sent(s.version).size < this.computeN) {
      if (!s.snap) this.asideNothing(svc);
      return;
    }
    const shown = this.shownSnap.forAside(), gen = shown?.gen ?? -1;
    if (this.asideDecided.get(svc) === gen) return;
    const left = unitsLeft(s, this.computeN, performance.now());
    if (left === null) { this.asideWaitsLeft = true; return; }
    this.asideDecided.set(svc, gen);
    const ent: AsideLog = { version: ++this.version, on: s.version, gen, held, at: performance.now(), mainMs: left };
    this.asideLog.push(ent);
    if (this.asideLog.length > 50) this.asideLog.shift();
    const no = (refused: string, why: string) => { this.asideEnd(ent, { refused, why }); this.asideNothing(svc); };
    const off = this.asideOff(), cities = this.asideCities(svc);
    // i18n-ignore (the dev log's)
    if (off) return no(off, off === "off" ? "?perfOff=editAside or focus" : off === "device" ? "a small or address-limited device" : off === "split" ? "?split=1" : this.asideFails >= ASIDE_FAILS ? `the side pool failed ${this.asideFails} times; none until another game` : "the side pool failed; none until the next snapshot");
    if (!shown) return no("snapshot", "no whole snapshot held"); // i18n-ignore
    if (!cities) return no("zones", "the network's reach is not known yet"); // i18n-ignore
    // (the fare test the plan makes, on the page: a fare edit never starts the pool to be refused there)
    const shownSvc = this.log.find((x) => x.version === demandView.peek()?.version)?.svc;
    if (shownSvc && JSON.stringify(shownSvc.fare) !== JSON.stringify(svc.fare)) return no("fares", "the fares changed since the figures on screen"); // i18n-ignore
    if (!this.asideFits([])) return no("memory", "the side pool and the snapshot would pass the memory budget"); // i18n-ignore
    if (ASIDE_GAIN * left <= (this.apool?.alive ? 0 : ASIDE_START_MS)) return no("slower", `the main pool's ${Math.round(left)} ms left`); // i18n-ignore
    this.asideJob = { svc, gen };
    void this.asideRun(ent, svc, s, held, shown, cities).catch((err) => no("error", String(err))).finally(() => {
      this.asideJob = null;
      this.asideIdle();
      this.asideSoon();
      this.refineAside();
    });
  }

  /** T-4220: a side job's plan, its zones opened and its solve, as `asideGo` decides (its results are not shown yet:
   * stage 7) */
  private async asideRun(ent: AsideLog, svc: Service, s: Solve, held: boolean, shown: AsideShown, cities: CityId[]) {
    const { version, gen } = ent;
    const no = (refused: string, why: string) => { this.asideEnd(ent, { refused, why }); this.asideNothing(svc); };
    const s1 = (ms: number) => `${(ms / 1000).toFixed(1)} s`; // i18n-ignore
    const pool = this.asidePool(), { net, ids } = packService(svc);
    const plan = await this.asideAsk(pool, (p) => p.aside({ kind: "aside", version, net, ids, rounds: ROUNDS, cities, gen, shown, ...(perfOff("periods") ? { fresh: true } : {}) }), (m) => (m.kind === "asidePlan" && m.version === version ? m.plan : undefined));
    ent.planMs = performance.now() - ent.at;
    if (!plan) return no("pool", "the side pool ended"); // i18n-ignore
    if (!plan.refused) this.sideFresh(svc, plan);
    if (plan.refused) return no(plan.refused, plan.why ?? "");
    ent.fresh = plan.fresh.map((z) => z.id);
    ent.periods = plan.periods;
    ent.expMs = plan.ms;
    if (!plan.fresh.length) return no("nothing", "no zone's view moved"); // i18n-ignore
    if (plan.ms === null) return no("time", "a fresh population's pass times are not known"); // i18n-ignore
    const level = plan.fresh.filter((z) => (this.levelPosted.get(z.id) ?? LOD_FINE) !== z.level);
    if (level.length) return no("level", `${level.map((z) => z.id).join(",")} held at another level now`); // i18n-ignore
    if (!this.asideFits(plan.fresh)) return no("memory", `with ${ent.fresh.join(",")} open there the side pool would pass the memory budget`); // i18n-ignore
    const openExp = this.asideOpenMs(plan.fresh), main = this.asideMainMs(s, held, plan);
    ent.openExp = openExp;
    ent.mainMs = main;
    if (main === null || !(plan.ms + openExp < ASIDE_GAIN * main)) return no("slower", `${s1(plan.ms + openExp)} against the main pool's ${main === null ? "not known" : s1(main)}`); // i18n-ignore
    // (the newest network and the newest snapshot win: a side job for an older one is never solved)
    const still = () => this.asideWanted()?.svc === svc && this.shownSnap.forAside()?.gen === gen;
    if (!still()) return this.asideEnd(ent, { superseded: true });
    const t0 = performance.now();
    if (!(await this.asideOpen(plan.fresh))) return no("pool", "the side pool ended while its zones opened"); // i18n-ignore
    ent.openMs = performance.now() - t0;
    const now = this.shownSnap.forAside();
    if (!still() || !now) return this.asideEnd(ent, { superseded: true });
    let results = 0;
    // (each cluster's periods, by its zones)
    const got = new Map<string, (Result | undefined)[]>();
    this.asideSeq = this.netSeq.get(svc) ?? -1;
    const done = await this.asideAsk(pool, (p) => p.asideSolve({ kind: "solve", version, net, ids, rounds: ROUNDS, ...(perfOff("chunks") ? { exactMarket: true } : {}), ...(perfOff("farChunks") ? { nearChunks: true } : {}), ...(perfOff("periods") ? { fresh: true } : {}), aside: { gen, cities, shown: now } }), (m) => {
      if (m.kind === "asideDone" && m.version === version) return m;
      if (m.kind === "result" && m.version === version && m.cluster?.aside) {
        results++;
        const k = m.cluster.cities.join();
        if (!got.has(k)) got.set(k, PERIODS.map(() => undefined));
        got.get(k)![m.period] = m;
      }
      return undefined;
    });
    if (!done) return no("pool", "the side pool ended while it solved"); // i18n-ignore
    ent.workerMs = done.ms;
    ent.results = results;
    ent.sums = done.sums;
    if (done.plan.refused) return no(done.plan.refused, done.plan.why ?? "");
    if (done.stale) return this.asideEnd(ent, { stale: true });
    const not = await this.asideShow(ent, svc, now, done.plan, got);
    this.asideEnd(ent, not ? { done: true, notShown: not } : { done: true });
    // (not shown with its network still the newest: none comes for it; a newer one's side job is decided next)
    if (not && this.latest() === svc) this.asideNothing(svc);
  }

  /**
   * T-4307: the side job decided for `svc` shows nothing and none comes for it (refused, failed, or not shown with `svc`
   * still the newest). Figures on screen from an older network's side job go: they may be of an edit the player has
   * undone or changed again since, and nothing would replace them before the main pool's result of `svc` (on a big
   * game a whole solve later). The main pool's figures show until then, as with no side job; T-2605's refine of an edit
   * of the zone in view may run for `svc` instead (`refineAside`). Figures of `svc`'s own network stay.
   */
  private asideNothing(svc: Service) {
    this.asideNone.add(svc);
    const o = this.side;
    if (o && (this.netSeq.get(svc) ?? -1) > o.seq) this.dropOverlay();
  }

  /** T-4220: a side job decided, its log entry ended (on the dev server, the console's line) */
  private asideEnd(ent: AsideLog, more: Partial<AsideLog>) {
    Object.assign(ent, more, { endAt: performance.now() });
    if (!import.meta.env.DEV) return;
    const s1 = (ms: number | null | undefined) => (ms === null || ms === undefined ? "not known" : `${(ms / 1000).toFixed(1)} s`); // i18n-ignore
    const head = `T4220 aside v${ent.version} (${ent.held ? "held behind" : "the network of"} v${ent.on}, snapshot gen ${ent.gen})`; // i18n-ignore
    const fresh = ent.fresh ? `${ent.fresh.join(",")} fresh, periods ${ent.periods?.join(",") || "none"}` : ""; // i18n-ignore
    // i18n-ignore
    console.info(ent.refused ? `${head}: refused, ${ent.refused}: ${ent.why}${fresh ? ` (${fresh})` : ""}` : ent.superseded ? `${head}: a newer network or snapshot came before its solve${fresh ? ` (${fresh})` : ""}` : `${head}: ${fresh}; ${ent.stale ? "stale (a newer job came first)" : `${ent.results} results, ${ent.notShown ? `not shown (${ent.notShown})` : `${ent.shown} clusters shown`}`}; ${s1(ent.endAt! - ent.at)} in all: plan ${s1(ent.planMs)}, opening ${s1(ent.openMs)}, worker ${s1(ent.workerMs)}; expected ${s1(ent.expMs)} and ${s1(ent.openExp)} opening against the main pool's ${s1(ent.mainMs)}`);
  }

  // --- T-4220 stage 7: a side job's figures shown over the main pool's ---

  /** a network's place in the order networks came in (`submitNetwork`), for "the side job's network or a newer one" */
  private netSeq = new WeakMap<Service, number>();
  private netCount = 0;
  /** T-4220 (notes/T-4220.md, "Stage 7"): the figures every publish merges into and writes, the main pool's (with T-2045's
   * and T-2605's put in). What is shown is these with a side job's over them (`sideOver`), so a side job's figures never
   * become the main pool's and go the moment they stop being wanted. */
  private mainDays: DayDemand[] = demandByDay.peek();
  private get days(): DayDemand[] {
    return this.mainDays;
  }
  private set days(days: DayDemand[]) {
    this.mainDays = days;
    // (T-4307: what is kept, money, points, achievements, stars and the boarding account, reads the main pool's alone)
    batch(() => {
      mainDemandByDay.value = days;
      demandByDay.value = this.sideOver(days);
    });
  }
  /** the side job's figures on screen, by cluster (`game`: the `gameLoadSeq` they are of; `removed`: lines on screen its
   * network no longer has), until the main pool shows its network or a newer one there (`sideCovered`), a later side
   * job's replace them, or a newer network's plan finds one of its fresh zones as on screen (`sideFresh`) */
  private side: { svc: Service; seq: number; game: number; version: number; removed: Set<string>; clusters: SideCluster[] } | null = null;
  /** per zone, the newest network (`netSeq`) a final of the main pool has been shown for; `coveredAll`: a whole final's */
  private coveredAt = new Map<CityId, number>();
  private coveredAll = -1;
  /** the network order of the last side job solved: once the main pool shows it or a newer one, the side pool ends */
  private asideSeq = Infinity;

  /**
   * T-4220 (notes/T-4220.md, "The pick" 6, "Stage 7"): a side job's figures put on screen over the main pool's, in place
   * of any earlier side job's. Not when its network is no longer the newest (an undo or another edit came meanwhile, which
   * the next side job covers), another game came, or the main pool has shown a final of its network or a newer one in the
   * cluster (a held edit's own local result, which races it once the running solve's final is in). Its lines read as
   * worked out (`linesSettled`); questions, the bubbles, the level of detail and "Calculating trips" stay the main
   * pool's. Resolves why nothing was shown, or "".
   */
  private asideShow(ent: AsideLog, svc: Service, shown: AsideShown, plan: AsideAnswer, got: Map<string, (Result | undefined)[]>): Promise<string> {
    const seq = this.netSeq.get(svc) ?? -1, game = gameLoadSeq.peek();
    const { net, layout } = packService(svc);
    return new Promise<string>((done) => this.inOrder(async () => {
      let why = "";
      try {
        // i18n-ignore (the dev log's)
        const gone = () => (game !== gameLoadSeq.peek() ? "another game" : this.latest() !== svc ? "a newer network came" : !this.days[0]?.view ? "nothing on screen" : "");
        if ((why = gone())) return;
        const off = new Uint32Array(net.lineN.length + 1);
        for (let l = 0; l < net.lineN.length; l++) off[l + 1] = off[l] + net.lineN[l];
        const wide = shown.flags.wide || shown.flags.whole;
        const clusters: SideCluster[] = [];
        for (const c of plan.clusters) {
          // (a side job's results name the cluster's fresh zones, `sideJob`)
          const res = got.get(c.fresh.join());
          if (!res || res.some((r) => !r) || this.coveredFrom(c.zones) >= seq) continue;
          const idx = Array.from(res[0]!.cluster!.lines), all = idx.map((l) => layout[l]).filter(Boolean);
          // (a city's population sees the cluster's lines with a stop it reaches, `reachView`; with `wide` or `whole`, all)
          const reach = wide ? [] : c.fresh.map((id) => cityReach(net, id, shown.zones.find((z) => z.id === id)?.box ?? null));
          const sees = (l: number) => wide || reach.some((r) => { for (let k = off[l]; k < off[l + 1]; k++) if (r[net.stops[k]]) return true; return false; });
          const seen = new Set(idx.filter(sees).map((l) => layout[l]?.id).filter((id): id is string => !!id));
          const ests = await this.daysOf({ version: ent.version, svc }, all, ROUNDS, true, res as Result[], true);
          clusters.push({ zones: c.zones, fresh: c.fresh, all, seen, ests });
        }
        if ((why = gone())) return;
        const keep = clusters.filter((c) => this.coveredFrom(c.zones) < seq);
        // i18n-ignore
        if (!keep.length) return void (why = clusters.length ? "the main pool showed its network first" : "no cluster whole");
        const ids = new Set(layout.map((l) => l.id));
        this.side = { svc, seq, game, version: ent.version, removed: new Set([...this.days[0].view!.lines.keys()].filter((id) => !ids.has(id))), clusters: keep };
        this.days = this.mainDays;
        const under = this.mainDays[0].view!;
        linesSettled(keep.flatMap((c) => (c.ests[0].view ? [...this.sideTakes(c, under, c.ests[0].view)] : [])), this.netWorld.get(svc) ?? -1);
        ent.shown = keep.length;
      } finally {
        done(why);
      }
    }));
  }

  /** `days` with the side job's figures over them: in each cluster the lines it takes (`sideTakes`) with their stations and
   * purpose entries, and its fresh zones' views; lines its network removed go. The totals, the market's figures, the other
   * zones, lines and stations, the bubbles and the view's version, round and final flag stay as `days` has them. */
  private sideOver(days: DayDemand[]): DayDemand[] {
    const o = this.side;
    if (!o) return days;
    if (o.game !== gameLoadSeq.peek()) {
      this.side = null;
      return days;
    }
    // (kinds of day standing in for another share their views: put over once)
    const made = new Map<DayDemand, { ests: DayDemand[]; out: DayDemand }[]>();
    return days.map((d, k) => {
      const under = d.view;
      if (!under) return d;
      const ests = o.clusters.map((c) => c.ests[k]);
      const was = made.get(d)?.find((m) => m.ests.every((e, i) => e === ests[i]));
      if (was) return was.out;
      const lines = new Map(under.lines), stations = new Map(under.stations);
      for (const id of o.removed) lines.delete(id);
      let purposes = under.purposes;
      const zones: [CityId, DemandView][] = [];
      o.clusters.forEach((c, i) => {
        const est = ests[i]?.view;
        if (!est) return;
        const take = this.sideTakes(c, under, est), at = new Set<string>();
        for (const l of c.all) if (take.has(l.id)) for (const id of [...l.stops, ...(l.back ?? [])]) if (est.stations.has(id)) at.add(id);
        for (const id of take) lines.set(id, est.lines.get(id)!);
        for (const id of at) stations.set(id, est.stations.get(id)!);
        if (est.purposes) {
          purposes = new Map(purposes);
          for (const [key, p] of est.purposes) {
            const old = purposes.get(key);
            const pl = new Map(old?.lines), ps = new Map(old?.stations);
            for (const id of take) { const l = p.lines.get(id); if (l) pl.set(id, l); }
            for (const id of at) { const v = p.stations.get(id); if (v) ps.set(id, v); }
            purposes.set(key, { lines: pl, stations: ps });
          }
        }
        for (const id of c.fresh) { const v = ests[i].cities.get(id); if (v) zones.push([id, v]); }
      });
      const view = { ...under, lines, stations, purposes };
      const cities = new Map<CityId, DemandView>();
      for (const [id, v] of d.cities) cities.set(id, { ...v, lines, stations, purposes });
      for (const [id, v] of zones) cities.set(id, { ...view, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay, trainShareByPeriod: v.trainShareByPeriod, split: v.split });
      const out: DayDemand = { view, cities, intercity: d.intercity };
      made.set(d, [...(made.get(d) ?? []), { ests, out }]);
      return out;
    });
  }

  /** the lines of side cluster `c` that take its figures over `under`'s: those a fresh zone's population sees, and those
   * `under` has no figures for on their stops, trains and cars (added or changed since); never one `under` shows at
   * other fares (T-3303) */
  private sideTakes(c: SideCluster, under: DemandView, est: DemandView): Set<string> {
    const out = new Set<string>();
    for (const l of c.all) {
      const got = est.lines.get(l.id);
      if (!got) continue;
      const was = under.lines.get(l.id);
      const fits = !!was && was.stops?.join() === got.stops?.join() && was.tph?.join() === got.tph?.join() && was.cars === got.cars;
      if ((c.seen.has(l.id) || !fits) && takesMarketless(was, got)) out.add(l.id);
    }
    return out;
  }

  /** T-4220: the main pool shows a final of `s` for `zones` (null: its whole final). The side job's figures there of
   * `s`'s network or an older one go; once it is the network of the side job in hand (with none, of the last one) or a
   * newer one, the side pool ends too (`asideIdle` keeps it through an older final, for an edit made during the next
   * whole solve). */
  private sideCovered(s: Solve, zones: readonly CityId[] | null) {
    const seq = this.netSeq.get(s.svc) ?? -1;
    if (!zones) this.coveredAll = Math.max(this.coveredAll, seq);
    else for (const id of zones) this.coveredAt.set(id, Math.max(this.coveredAt.get(id) ?? -1, seq));
    const o = this.side;
    if (o && seq >= o.seq) {
      const z = zones && new Set(zones);
      o.clusters = z ? o.clusters.filter((c) => !c.zones.some((id) => z.has(id))) : [];
      for (const l of s.layout) o.removed.delete(l.id);
      if (!o.clusters.length) this.side = null;
    }
    // (T-4307: the job in hand's figures could then never be shown: a hung or slow one ends here too)
    const job = this.asideJob;
    if (seq >= (job ? this.netSeq.get(job.svc) ?? -1 : this.asideSeq)) this.endAside();
  }
  /** the newest network a final of the main pool has been shown for in any of `zones` */
  private coveredFrom(zones: readonly CityId[]): number {
    return Math.max(this.coveredAll, ...zones.map((id) => this.coveredAt.get(id) ?? -1));
  }

  /** T-4220: a newer network's plan. A cluster of the side job's figures on screen with a fresh zone the plan finds as on
   * screen (the edit there undone) is of an edit the player no longer has: it goes, and the main pool's figures show */
  private sideFresh(svc: Service, plan: AsideAnswer) {
    const o = this.side;
    if (!o || (this.netSeq.get(svc) ?? -1) <= o.seq) return;
    const fresh = new Set(plan.fresh.map((z) => z.id));
    const keep = o.clusters.filter((c) => c.fresh.every((id) => fresh.has(id)));
    if (keep.length === o.clusters.length) return;
    o.clusters = keep;
    if (!keep.length) this.side = null;
    this.days = this.mainDays;
  }

  /** T-4220: another game: the side job's figures and what the main pool has shown go */
  private dropSide() {
    this.coveredAt.clear();
    this.coveredAll = -1;
    this.asideSeq = Infinity;
    this.asideFails = 0;
    this.dropOverlay();
  }

  /** the side job's figures leave the screen (only those: what the main pool has shown stays known, `coveredFrom`) */
  private dropOverlay() {
    if (!this.side) return;
    this.side = null;
    this.days = this.mainDays;
  }

  /** T-4220: the side pool, started if it is not running (T-2605's refine aside ends: its zone's figures stay shown until
   * the next result) */
  private asidePool(): FocusPool {
    if (this.apool?.alive) return this.apool;
    this.endFocus();
    this.alevel.clear();
    return (this.apool = new FocusPool(1, [periodsOfWorker(0, 1)], {
      message: (_, msg) => this.atake?.(msg),
      opened: (id) => { this.aopening.get(id)?.(); this.aopening.delete(id); },
      // i18n-ignore
      failed: (why) => { console.warn(`demand: the side pool stopped (${why})`); this.lastTrouble = `side pool stopped: ${why}`; this.asideBroken = true; this.asideFails++; this.endAside(); },
    }, this.viewMsg(), false, ASIDE_MAX_ZONES));
  }

  /** T-4220: the side pool's workers go; an exchange or an opening waiting on it ends */
  private endAside() {
    const p = this.apool;
    if (!p) return;
    this.apool = null;
    p.end();
    this.alevel.clear();
    for (const r of this.aopening.values()) r();
    this.aopening.clear();
    this.atake?.(null);
    // (T-2605's relevel aside waited for it)
    queueMicrotask(() => this.refineAside());
  }

  /** T-4220: the side pool ends unless a whole solve runs that an edit could still be made during and it holds a zone */
  private asideIdle() {
    const p = this.apool, s = this.cur;
    if (!p || this.asideJob) return;
    if (!p.open.length || !s || s.barrier !== false || s.rough || s.published >= ROUNDS) this.endAside();
  }

  /** an exchange with the side pool: `send`, then each of its messages to `take` until it answers (null: the pool ended) */
  private asideAsk<T>(pool: FocusPool, send: (p: FocusPool) => void, take: (msg: FromDemand) => T | undefined): Promise<T | null> {
    if (this.apool !== pool || !pool.alive) return Promise.resolve(null);
    this.atake?.(null); // (one at a time: an earlier exchange still waiting is answered as ended, T-4297)
    return new Promise<T | null>((done) => {
      this.atake = (msg) => {
        const v = msg ? take(msg) : null;
        if (v === undefined) return;
        this.atake = null;
        done(v);
      };
      send(pool);
    });
  }

  /** T-4220: `zones` open in the side pool at their levels (false: it ended meanwhile, or a pack did not download). Past its
   * cap the oldest of its other zones close first, so the pool never closes one of these to open another */
  private async asideOpen(zones: { id: CityId; level: number }[]): Promise<boolean> {
    const pool = this.apool;
    if (!pool?.alive) return false;
    const want = new Set(zones.map((z) => z.id));
    const go = zones.filter(({ id, level }) => !(pool.open.includes(id) && this.alevel.get(id) === level));
    for (const { id } of go) if (pool.holds(id)) pool.close(id);
    for (const id of [...pool.open, ...pool.opening]) if (!want.has(id) && pool.open.length + pool.opening.size + go.length > ASIDE_MAX_ZONES) pool.close(id);
    try {
      await Promise.all(go.map(async ({ id, level }) => {
        const [header, bin] = await this.downloadPack(id);
        if (this.apool !== pool) return;
        this.aopening.get(id)?.();
        const opened = new Promise<void>((r) => this.aopening.set(id, r));
        this.alevel.set(id, level);
        const own = bin.byteOffset === 0 && bin.byteLength === bin.buffer.byteLength ? bin.buffer as ArrayBuffer : bin.slice().buffer;
        pool.post({ id, header: new TextDecoder().decode(header), bin: own, level });
        await opened;
      }));
    } catch (err) {
      console.warn(`demand: a side job's zone did not download (${String(err)})`); // i18n-ignore
      return false;
    }
    return this.apool === pool && zones.every(({ id }) => pool.open.includes(id));
  }

  /** T-4220: the ms to open `zones` in the side pool where not open there at their levels, as the load's model has a city
   * pack (`OPEN_CITY_MS`, `OPEN_MS_PER_MB`), one after another in its one worker */
  private asideOpenMs(zones: { id: CityId; level: number }[]): number {
    const p = this.apool;
    return zones.reduce((ms, { id, level }) => (p?.open.includes(id) && this.alevel.get(id) === level ? ms : ms + OPEN_CITY_MS + OPEN_MS_PER_MB.city * (this.bytes.get(id) ?? GUESS_BYTES) / 1e6), 0);
  }

  /** T-4220: the main pool's expected ms to the edit's lines: the running whole solve's time left, and for a network held
   * behind it, its local stage after that, the plan's periods dealt over the solving workers as the main pool deals them */
  private asideMainMs(s: Solve, held: boolean, plan: AsideAnswer): number | null {
    const left = unitsLeft(s, this.computeN, performance.now());
    if (left === null || !held || plan.ms === null || !plan.periods.length) return left;
    const most = Math.max(...this.periodsOf.slice(0, this.computeN).map((qs) => plan.periods.filter((q) => qs.includes(q)).length));
    return left + plan.ms * most / plan.periods.length;
  }

  /** T-4220: the side pool with `zones` open there too fits the memory budget as `memShort` counts the page: a worker's JS
   * (`MEM_WORKER`), its WASM (as last reported, else a start's), the snapshot copied there, and each zone not open there at
   * its level by `zoneGB` */
  private asideFits(zones: { id: CityId; level: number }[]): boolean {
    const p = this.apool?.alive ? this.apool : null;
    let add = MEM_WORKER + this.shownSnap.bytes() + (p?.mem.length ? p.mem.reduce((a, b) => a + (b ?? 0), 0) : MEM_BASE);
    for (const { id, level } of zones) if (!(p?.open.includes(id) && this.alevel.get(id) === level)) add += zoneGB(id, level === LOD_COARSE) * 2 ** 30;
    return this.memHeld() + add <= GROW_ROOM * memBudget();
  }

  /** the newest network and the snapshot a side job's dev call goes against, or null (logged) */
  private devAsideOn(what: string) {
    const svc = this.latest(), shown = this.shownSnap.forAside(), cities = svc && this.asideCities(svc);
    // i18n-ignore
    if (!svc || !shown || !cities || this.asideJob) { console.info(`T4220 ${what}: ${this.asideJob ? "a side job is in hand" : "no whole snapshot held"}`); return null; }
    return { ...packService(svc), svc, shown, version: ++this.version, cities };
  }

  /**
   * T-4220 stage 3, dev server only (`?debug=1`, `twDemand.client.asidePlan()`): the side job's plan for the newest
   * network against the snapshot held, in the side pool, logged and returned.
   */
  async asidePlan(): Promise<AsideAnswer | null> {
    if (!import.meta.env.DEV) return null;
    const on = this.devAsideOn("aside plan"); // i18n-ignore
    if (!on) return null;
    this.asideJob = { svc: on.svc, gen: on.shown.gen };
    try { return await this.devPlan(on); } finally { this.asideJob = null; this.asideIdle(); }
  }
  private async devPlan(on: NonNullable<ReturnType<DemandClient["devAsideOn"]>>): Promise<AsideAnswer | null> {
    const { net, ids, cities, version, shown } = on, t0 = performance.now();
    const plan = await this.asideAsk(this.asidePool(), (p) => p.aside({ kind: "aside", version, net, ids, rounds: ROUNDS, cities, gen: shown.gen, shown, ...(perfOff("periods") ? { fresh: true } : {}) }), (m) => (m.kind === "asidePlan" && m.version === version ? m.plan : undefined));
    if (!plan) return null;
    const qs = (m: number) => PERIODS.map((_, q) => q).filter((q) => m & (1 << q)).join(",") || "none";
    const s = (ms: number | null) => (ms === null ? "not known" : `${(ms / 1000).toFixed(1)} s`); // i18n-ignore
    // i18n-ignore
    console.info(plan.refused ? `T4220 aside plan v${version} (snapshot ${shown.pops.length} populations): refused, ${plan.refused}: ${plan.why}` : `T4220 aside plan v${version} (snapshot ${shown.pops.length} populations): ${plan.fresh.length ? plan.fresh.map((z) => `${z.id}${z.level === LOD_COARSE ? " coarse" : ""}`).join(",") : "no zone"} fresh of ${shown.zones.length}; ${plan.clusters.map((c) => `cluster ${c.zones.slice(0, 6).join(",")}${c.zones.length > 6 ? ",..." : ""}: periods ${c.solve.join(",") || "none"} solved, ${qs(c.keep)} kept, ${s(c.ms)}`).join("; ") || "nothing to solve"}; expected ${s(plan.ms)}; planned in ${Math.round(plan.planMs)} ms, answered in ${Math.round(performance.now() - t0)} ms`);
    return plan;
  }

  /**
   * T-4220 stage 4, dev server only (`?debug=1`, `twDemand.client.aside()`): a side job for the newest network against the
   * snapshot held, in the side pool: planned, its fresh zones opened there at the level the snapshot's solve held them
   * (refused when the main pool holds one at another level now), then solved, whatever the start rule would say. Logs
   * `T4220 aside ...` and returns the plan, its end (`asideDone`), its results by period and bubble sums, and from those
   * results by the page's own reduction (`dayViews`, the weekday) each of its clusters' lines' riders and stations'
   * boardings, to set beside a main-pool solve of the same network.
   */
  async aside(): Promise<{ plan: AsideAnswer; done?: Extract<FromDemand, { kind: "asideDone" }>; results?: Result[]; sums?: Extract<FromDemand, { kind: "sums" }>[]; lines?: Record<string, number>; stations?: Record<string, number>; openMs?: number; ms?: number } | null> {
    if (!import.meta.env.DEV) return null;
    const on = this.devAsideOn("aside"); // i18n-ignore
    if (!on) return null;
    this.asideJob = { svc: on.svc, gen: on.shown.gen };
    try { return await this.devAside(on); } finally { this.asideJob = null; this.asideIdle(); }
  }
  private async devAside(on: NonNullable<ReturnType<DemandClient["devAsideOn"]>>) {
    const t0 = performance.now(), plan = await this.devPlan(on);
    if (!plan || plan.refused) return plan && { plan };
    const { net, ids, layout, cities, version, shown } = on;
    const off = plan.fresh.filter((z) => (this.levelPosted.get(z.id) ?? LOD_FINE) !== z.level);
    // i18n-ignore
    if (off.length) { console.info(`T4220 aside v${version}: refused, level: ${off.map((z) => z.id).join(",")} held at another level now`); return { plan: { ...plan, refused: "level" as const, why: off.map((z) => z.id).join(",") } }; }
    const tOpen = performance.now(), pool = this.apool;
    if (!pool || !(await this.asideOpen(plan.fresh))) return { plan };
    const openMs = performance.now() - tOpen, results: Result[] = [], sums: Extract<FromDemand, { kind: "sums" }>[] = [];
    const done = await this.asideAsk(pool, (p) => p.asideSolve({ kind: "solve", version, net, ids, rounds: ROUNDS, ...(perfOff("chunks") ? { exactMarket: true } : {}), ...(perfOff("farChunks") ? { nearChunks: true } : {}), ...(perfOff("periods") ? { fresh: true } : {}), aside: { gen: shown.gen, cities, shown } }), (m) => {
      if (m.kind === "asideDone") return m;
      if (m.kind === "result" && m.cluster?.aside) results.push(m);
      else if (m.kind === "sums" && m.aside) sums.push(m);
      return undefined;
    });
    if (!done) return { plan };
    // each cluster's five periods, its lines' riders and stations' boardings a weekday as the page works them out
    const lines: Record<string, number> = {}, stations: Record<string, number> = {};
    const byCluster = new Map<string, Result[]>();
    for (const r of results) { const k = r.cluster!.cities.join(); byCluster.set(k, [...(byCluster.get(k) ?? []), r]); }
    for (const rs of byCluster.values()) {
      const res = PERIODS.map((_, q) => rs.find((r) => r.period === q));
      if (res.some((r) => !r)) continue;
      const own = new Set(rs[0].cluster!.lines);
      const [d] = dayViews({ version, round: ROUNDS, final: true, layout: layout.filter((_, i) => own.has(i)), res: res.map((r) => dayResultOf(r!)), dayTable: this.dayTable, peakHourFactor: PERIODS.map((_, q) => this.periodTable[q * PERIOD_STRIDE + 5] ?? 1), solved: [], places: INTERCITY_PLACES.length, days: [0], noCorridors: true });
      for (const [id, l] of d.view.lines) lines[id] = l.ridersPerDay;
      for (const [id, st] of d.view.stations) stations[id] = st.boardings;
    }
    const s1 = (ms: number) => `${(ms / 1000).toFixed(1)} s`; // i18n-ignore
    const solved = [...new Set(results.map((r) => r.period))].sort().join(",");
    // i18n-ignore
    console.info(done.plan.refused ? `T4220 aside v${version}: refused, ${done.plan.refused}: ${done.plan.why}` : done.stale ? `T4220 aside v${version}: stale (a newer job came first)` : `T4220 aside v${version} (snapshot gen ${shown.gen}): ${plan.fresh.map((z) => z.id).join(",")} fresh of ${shown.zones.length} zones; ${results.length} results (periods ${solved}); ${Object.keys(lines).length} lines, ${Math.round(Object.values(lines).reduce((a, b) => a + b, 0))} riders; bubble sums of ${done.sums?.join(",") || "none"}; ${s1(done.ms)} in the worker, expected ${plan.ms === null ? "not known" : s1(plan.ms)}, opening ${s1(openMs)}, ${s1(performance.now() - t0)} in all`);
    return { plan, done, results, sums, lines, stations, openMs, ms: performance.now() - t0 };
  }

  /** T-1174, dev server only (`?debug=1`, `twDemand.client.debugPanic`): make worker `i` panic. */
  debugPanic(i = 0, when: "now" | "solve" = "now", oom = false) {
    if (import.meta.env.DEV) this.workers[i]?.postMessage({ kind: "debugPanic", when, oom } satisfies ToDemand);
  }

  // --- T-1205: memory. An idle worker well above its need restarts; big games run fewer workers ---

  /** each worker's WASM memory and heap as last reported; `solvePeak`, its heap's most in its last solve */
  private mem: { wasm: number; heap: Heap | null; solvePeak: number }[] = [];
  /** each worker's WASM and need after its last restart's solve (`overNeed`'s guard against restarting
   * again for nothing) */
  private restarted: ({ wasm: number; need: number } | undefined)[] = [];
  /** the pool's size at the start: the demand cache's key (a shrink changes no figure) */
  private poolN = 0;
  /** when the newest network arrived or a solve's final was published, performance.now() */
  private lastStir = 0;
  private memTimer = 0;
  /** T-2188: when the player's last edit reached the pool (`submitNetwork`), performance.now(); city edits ended at
   * their local result since the last solve that left no local result anywhere; the refresh's timer */
  private editAt = 0;
  private localEdits = 0;
  private refreshTimer = 0;
  /** solving workers that have sent `done` for the solve in hand */
  private doneBy = new Set<number>();
  /** a solve after a worker change (`freshParts`) has not published its final yet: questions wait
   * for it, since the new worker holds no earlier result (`target`) */
  private refreshing = false;
  /** the page's memory outside the workers' WASM, from `measureUserAgentSpecificMemory` (null where
   * that is not available: not a cross-origin isolated page) */
  private pageOther: number | null = null;

  /** T-3757: the page outside the demand workers' WASM where it cannot be measured, which grows with the network: on a
   * 7,000-station save about 5 GB, far over MEM_OTHER, so a pool of more cores ran past the budget */
  private otherGuess(): number {
    let first = 0;
    for (let i = 0; i < this.computeN; i++) first = Math.max(first, this.mem[i]?.wasm ?? 0);
    return Math.max(MEM_OTHER, outsideWasmBytes(first, this.mem[this.computeN]?.wasm ?? 0));
  }

  /** `solved`: after a solve, whose heap peak stands until something closes (it held what closed) */
  private noteMem(i: number, wasm: number, heap: Heap | undefined, solved = false, closed = false) {
    const m = (this.mem[i] ??= { wasm: 0, heap: null, solvePeak: 0 });
    m.wasm = wasm;
    if (closed) m.solvePeak = 0;
    if (heap) { m.heap = heap; if (solved) m.solvePeak = heap[1]; }
  }

  /** About the WASM worker `i` needs for what it holds open and a solve, as a worker started afresh
   * reaches: its heap at rest with what a solve adds, or its last solve's heap peak, whichever is
   * more, with the allocator's slack. Null before it has said. */
  private needOf(i: number): number | null {
    const m = this.mem[i];
    return m?.heap ? Math.max(m.heap[0] * MEM_NEED[0], m.solvePeak * MEM_NEED[1]) + MEM_BASE : null;
  }

  /** The workers above their need by MEM_OVER and MEM_FREE; not one restarted since when neither its
   * memory nor its need has moved by MEM_FREE (a restart would give nothing back). */
  private overNeed(): { i: number; wasm: number; need: number }[] {
    const out: { i: number; wasm: number; need: number }[] = [];
    for (let i = 0; i < this.workers.length; i++) {
      const m = this.mem[i], need = this.needOf(i), last = this.restarted[i];
      if (!m || need === null || m.wasm < need * MEM_OVER || m.wasm - need < MEM_FREE) continue;
      if (last && m.wasm < last.wasm + MEM_FREE && need > last.need - MEM_FREE) continue;
      out.push({ i, wasm: m.wasm, need });
    }
    return out;
  }

  /** T-4204: what the workers hold now (their WASM, not their need) with the rest of the page passes GROW_ROOM of the
   * budget, the line the pool grows to. The share of the solving workers (`SOLVING_SHARE`) is not asked: it sizes the
   * pool by need, which a restart does not change. */
  private memShort(): boolean {
    return this.memHeld() > GROW_ROOM * memBudget();
  }
  /** what `memShort` counts the page as holding: the rest of the page, and each worker's WASM and JS */
  private memHeld(): number {
    const js = this.pageOther === null ? MEM_WORKER : 0;
    // (the rest of the page grows with the network, which a worker's need stands for: its WASM over that is the
    // allocator's high water, which `otherGuess` would count again outside)
    const size = (i: number) => Math.min(this.mem[i]?.wasm ?? 0, this.needOf(i) ?? Infinity);
    let first = 0;
    for (let i = 0; i < this.computeN; i++) first = Math.max(first, size(i));
    let held = this.pageOther ?? Math.max(MEM_OTHER, outsideWasmBytes(first, size(this.computeN)));
    for (let i = 0; i < this.workers.length; i++) held += (this.mem[i]?.wasm ?? 0) + js;
    return held;
  }

  /** a solve's rounds are still out with the solving workers */
  private solving(): boolean {
    const s = this.cur;
    return !!s && (s.published < ROUNDS || this.doneBy.size < this.computeN);
  }

  /** ms until nothing is in flight and the network has been still for MEM_QUIET_MS (0: now). Never
   * during a solve, a load, an open or a question, nor while a worker is being replaced. */
  private quietIn(): number {
    const s = this.cur;
    if (this.status !== "ready" || this.respawn.size || this.waiting || this.job || this.hold || this.ready < this.workers.length || this.pendingQueries.size) return MEM_RETRY_MS;
    if (s && (s.published < ROUNDS || this.doneBy.size < this.computeN)) return MEM_RETRY_MS;
    if (this.fetchingUndecided.size || [...this.fetching].some((x) => !this.opened.has(x)) || [...this.fetchingBare].some((x) => !this.bareOpen.has(x))) return MEM_RETRY_MS;
    return Math.max(0, this.lastStir + MEM_QUIET_MS - performance.now());
  }

  private memSoon(ms = MEM_QUIET_MS) {
    clearTimeout(this.memTimer);
    this.memTimer = window.setTimeout(() => this.memCheck(), ms);
  }

  /** Restart the workers well above their need, together, once the pool is quiet, and add back the solving
   * workers that now fit (T-1366, `growTo`). The replacements and new workers open what the pool holds and the
   * network is solved again once (`workerDied`'s path); the figures shown stay until that solve's final, the
   * same to the trip. T-4204: a restart alone (no growth) waits until the pool is short of memory (`memShort`). */
  private memCheck() {
    this.memTimer = 0;
    if (this.failed) return;
    // T-3757: the budget's cut once quiet after a final too, so a big save opened with more cores does not hold
    // every worker until the next solve; the workers left hold periods they never solved, so the network is solved
    // again as after a growth
    if (this.computeN > 1) {
      const wait = this.quietIn();
      if (wait > 0) return this.memSoon(wait);
      const n0 = this.computeN;
      this.capPool();
      if (this.computeN < n0) return this.solveAgain();
    }
    const over = this.overNeed();
    const n0 = this.computeN, n = this.growTo(new Set(over.map((o) => o.i)));
    if (!over.length && n <= n0) return;
    const wait = this.quietIn();
    if (wait > 0) return this.memSoon(wait);
    const gap = this.poolChangedAt + this.growGap - performance.now();
    // T-4204: a replacement holds nothing solved, so the whole network is solved again (4-5 minutes of "calculating
    // trips" on a 140-zone Asia save, for figures that do not change): a worker's slack above its need stays until the
    // pool is short of memory, or a growth solves everything again anyway
    if (!(n > n0 && gap <= 0) && !this.memShort() && !perfOff("keepSlack")) {
      if (over.length) console.info(`demand: workers kept over their need, the pool not short of memory: ${over.map((o) => `${o.i} ${mb(o.wasm)} MB of WASM, about ${mb(o.need)} MB needed`).join("; ")}`); // i18n-ignore
      if (n > n0) this.memSoon(gap);
      return;
    }
    if (n > n0 && gap > 0) this.memSoon(gap);
    else if (n > n0) this.grow(n, `about ${mb(this.growPer)} MB a solving worker, room for ${n} in ${mb(GROW_ROOM * this.solvingRoom())} MB`); // i18n-ignore
    // (the aux worker has moved to the end of the grown pool)
    for (const o of over) this.workerDied(o.i === n0 ? this.computeN : o.i, `${mb(o.wasm)} MB of WASM, about ${mb(o.need)} MB needed`, true); // i18n-ignore
  }

  /** T-1366: when the pool last grew or was cut (performance.now()), and the wait after it before it may grow */
  private poolChangedAt = 0;
  private growGap = GROW_GAP_MS;
  private grew = false;
  /** `growTo`'s last count of a solving worker, for the console */
  private growPer = 0;
  /** what the solving workers may take together: `SOLVING_SHARE` of the budget, or all of it with the share off */
  private solvingRoom(): number {
    return shareCapped() ? solvingShare() * memBudget() : memBudget();
  }

  /** T-1366: the solving workers the pool may have now, `computeN` or more, up to its size at the start
   * (`poolN`; T-3072's remembered cap, `poolSize`). Each counts as the most any of them holds now (its WASM)
   * or needs, a new worker opening all they hold; `restarting`, about to restart, as its need. Room: GROW_ROOM
   * of `capPool`'s two limits, so a pool grown is a step away from being cut again. */
  private growTo(restarting: Set<number>): number {
    const n0 = this.computeN, max = Math.min(this.poolN, poolSize());
    // (T-3373 B: under `?split=1` a grown worker would hold nothing; dealing to it is stage 7's)
    if (n0 >= max || this.respawn.size || perfOff("cap") || perfOff("grow") || workersAsked() !== null || this.holding) return n0;
    const held = (i: number) => { const need = this.needOf(i); return need === null ? null : restarting.has(i) ? need : Math.max(need, this.mem[i]?.wasm ?? 0); };
    let per = 0;
    for (let i = 0; i < n0; i++) { const h = held(i); if (h === null) return n0; per = Math.max(per, h); }
    this.growPer = per;
    const js = this.pageOther === null ? MEM_WORKER : 0, other = this.pageOther ?? this.otherGuess();
    const aux = (held(n0) ?? 0) + js, budget = memBudget();
    const share = shareCapped() ? solvingShare() * budget : Infinity;
    let n = n0;
    while (n < max && other + (n + 1) * (per + js) + aux <= GROW_ROOM * budget && (n + 1) * (per + MEM_WORKER) <= GROW_ROOM * share) n++;
    return n;
  }

  /** T-1366: `shrink` in reverse: solving workers `computeN` to `n` - 1 start, open what the pool holds as a
   * replacement does (`reopen`), and the periods are dealt again among all; the aux worker moves up to index `n`.
   * Once every new worker holds it all, the network is solved again (`solveAgain`). */
  private grow(n: number, why: string) {
    const old = this.computeN, add = n - old;
    console.info(`demand: ${old} solving workers to ${n}: ${why}`);
    for (const list of [this.workers, this.ackedBy, this.start.workers, this.mem, this.restarted] as unknown[][]) {
      while (list.length < old + 1) list.push(undefined);
      list.splice(old, 0, ...new Array(add).fill(undefined));
    }
    if (this.isReady.delete(old)) this.isReady.add(n);
    this.computeN = n;
    this.dealPeriods(n);
    solveWaves = Math.ceil(PERIODS.length / n);
    this.periodsOf.forEach((periods, i) => { if (i < old || i === n) this.workers[i].postMessage({ kind: "periods", periods, ...(i < n ? this.marketsMsg(i) : {}), ...(i === n ? { groups: this.auxGroups() } : {}) } satisfies ToDemand); });
    for (let i = old; i < n; i++) {
      this.respawn.set(i, { left: null, solving: false, planned: true });
      this.workers[i] = this.spawn(i, false);
    }
    this.endQueries();
    this.freshParts = true;
    this.poolChangedAt = performance.now();
    this.grew = true;
  }

  /** Before a solve, and as zones open while none runs: fewer solving workers when together they would
   * pass `SOLVING_SHARE` of the budget (`memBudget`), or with the rest of the page the budget itself. Each
   * holds every open zone and the market, so a worker less saves its whole need, at a slower solve; no
   * figure changes. T-1366: `memCheck` grows it back (`growTo`). */
  private capPool() {
    if (this.computeN <= 1 || this.respawn.size || perfOff("cap") || workersAsked() !== null) return;
    let per = 0, sum = 0;
    for (let i = 0; i < this.computeN; i++) { per = Math.max(per, this.needOf(i) ?? 0); sum += this.needOf(i) ?? 0; }
    if (!per) return;
    // (a measured page counts each worker's JS heap in `pageOther`)
    const js = this.pageOther === null ? MEM_WORKER : 0, other = this.pageOther ?? this.otherGuess();
    const aux = (this.needOf(this.computeN) ?? 0) + js, budget = memBudget();
    let n = this.computeN;
    const share = shareCapped() ? solvingShare() * budget : Infinity;
    // (T-3373 B, `?split=1`: a cut moves the stopped workers' units to the rest, which then hold what all held: a
    // worker less saves only its JS heap)
    const pool = (k: number, perJs: number) => (this.holding ? sum + k * perJs : k * (per + perJs));
    while (n > 1 && (other + pool(n, js) + aux > budget || pool(n, MEM_WORKER) > share)) n--;
    // i18n-ignore
    if (n < this.computeN) { this.shrink(n, `about ${mb(per + js)} MB a solving worker, ${mb(aux)} MB the aux worker, ${mb(other)} MB the rest of the page, against ${mb(budget)} MB`); noteLoadWorkers(n); }
  }

  /** T-3373: a loading save's pool cut to what workers of `perWorker` bytes may be (`workersFor`), before
   * its packs open; `capPool` judges again by what they then hold. T-3757: by the budget too, the rest of the page
   * guessed as `capPool` does (a 7,000-station save otherwise loaded on 5 workers, 13 GB). T-4263: `capPool`'s own
   * test, the aux worker (`auxWasm` bytes of WASM) counted as it counts it: a pool it would cut after the load's
   * first final is solved again whole on the rest, every question waiting for that (minutes on a big save). */
  private capAhead(perWorker: number, auxWasm: number) {
    if (this.computeN <= 1 || this.respawn.size || perfOff("cap") || workersAsked() !== null || this.solving()) return;
    const aux = auxWasm * MEM_NEED[0] + MEM_BASE + MEM_WORKER;
    const byBudget = poolIn(perWorker, this.computeN, memBudget() - Math.max(MEM_OTHER, outsideWasmBytes(perWorker - MEM_WORKER, auxWasm)) - aux);
    const n = Math.min(this.computeN, byBudget, shareCapped() ? workersFor(perWorker) : Infinity);
    if (n < this.computeN) { this.shrink(n, `about ${mb(perWorker)} MB a solving worker by the save's zones`); noteLoadWorkers(n); } // i18n-ignore (T-3933: a planned cut is what the load runs)
  }

  /** The pool's last solving workers stop and the periods are dealt again among the rest (between
   * solves in each worker); the aux worker moves down to index `n`. */
  private shrink(n: number, why: string) {
    const old = this.computeN, cut = old - n;
    console.info(`demand: ${old} solving workers to ${n}: ${why}`);
    this.poolChangedAt = performance.now();
    if (this.grew) this.growGap *= 2; // (T-1366: grown too soon)
    for (let k = n; k < old; k++) this.workers[k].terminate();
    for (const list of [this.workers, this.ackedBy, this.start.workers, this.mem, this.restarted] as unknown[][]) list.splice(n, cut);
    const auxReady = this.isReady.has(old);
    this.isReady = new Set([...this.isReady].filter((i) => i < n));
    if (auxReady) this.isReady.add(n);
    this.ready = this.isReady.size;
    this.acks.clear();
    for (const by of this.ackedBy) for (const item of by ?? []) this.acks.set(item, (this.acks.get(item) ?? 0) + 1);
    this.computeN = n;
    // (T-2867: in a load, what only the stopped workers were still opening is now open everywhere)
    for (const [item, count] of this.acks) if (count >= this.holdersOf(item).length && !this.opened.has(item) && !this.broken.has(item)) this.openEverywhere(item);
    this.dealPeriods(n);
    solveWaves = Math.ceil(PERIODS.length / n);
    this.periodsOf.forEach((periods, i) => this.workers[i].postMessage({ kind: "periods", periods, ...(i < n ? this.marketsMsg(i) : {}), ...(i === n ? { groups: this.auxGroups() } : {}) } satisfies ToDemand));
    this.endQueries();
    this.freshParts = true;
    // T-3373 B: what only the stopped workers held goes to a survivor, downloaded again; the next solve waits for it
    for (const unit of this.holding?.shrink(n) ?? []) {
      const items = unit === MARKET_UNIT ? [...this.fetching].filter(isMarket) : this.fetching.has(unit) ? [unit] : [];
      for (const item of items) { this.opened.delete(item); this.unack(item); this.packAcks.delete(item); }
      if (unit !== MARKET_UNIT) { if (items.length) this.fetchCity(unit as CityId, "full"); continue; }
      this.rows = null;
      this.rowsSent = false;
      for (const item of items) { this.fetching.delete(item); this.fetchMarket(marketZone(item)); }
    }
    // T-3868: only a game already solved is solved again; a load's cut (`capAhead` on every big save) frees nothing
    if (!this.hold && !gameLoadArmed() && !this.cacheShown() && demandView.peek()?.final) memoryResolve.value = true;
  }

  /** T-1205: the page's memory outside the workers' WASM, where the browser can say (a cross-origin
   * isolated page); asked after a final, as it may take until the next garbage collection. */
  private async measurePage() {
    const p = performance as { measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }> };
    if (!p.measureUserAgentSpecificMemory || !(globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated) return;
    try {
      const { bytes } = await p.measureUserAgentSpecificMemory();
      this.pageOther = Math.max(0, bytes - this.mem.reduce((a, m) => a + (m?.wasm ?? 0), 0));
    } catch { /* not allowed here */ }
  }

  /** Dev server only (`?debug=1`, `twDemand.client.debugRestart`): restart workers as `memCheck` does. */
  debugRestart(...workers: number[]) {
    if (import.meta.env.DEV) for (const i of workers) this.workerDied(i, "a test restart", true); // i18n-ignore
  }

  /** Each worker's memory as last reported, MB (`twDemand.client.memory()`, checks). */
  memory() {
    return this.mem.map((m, i) => ({ worker: i, aux: i === this.computeN, wasm: mb(m?.wasm ?? 0), heap: mb(m?.heap?.[0] ?? 0), peakNow: mb(m?.heap?.[1] ?? 0), solvePeak: mb(m?.solvePeak ?? 0), need: mb(this.needOf(i) ?? 0) }));
  }

  /** A question's answer (from either pool): true if `msg` is one. */
  private reply(i: number, msg: FromDemand): msg is Reply {
    if (!isQuery(msg)) return false;
    const w = this.pendingQueries.get(msg.id);
    if (!w) return true;
    const t = msg as { qStart?: number; qMs?: number };
    this.qlog.find((q) => q.id === msg.id)?.replies.push({ w: i, start: t.qStart ?? 0, ms: t.qMs ?? 0, at: performance.timeOrigin + performance.now() });
    w.replies.push(msg);
    if (--w.left === 0) {
      this.pendingQueries.delete(msg.id);
      w.done(w.replies);
      // (T-2533: a moment for the questions that follow from this one, a far end's spread after its sums)
      if (w.focus && this.cacheShown()) { clearTimeout(this.retireTimer); this.retireTimer = window.setTimeout(() => this.retireFocus(), 1500); }
    }
    return true;
  }

  // --- T-2045: the zone in view solved alone, to its last round, while a game load solves ---------

  /** the focus pool of the game loading (`demandFocus.ts`), null outside a load or with ?perfOff=focus */
  private fpool: FocusPool | null = null;
  /** its solve in hand */
  private fsolve: Solve | null = null;
  /** per zone, the focus solve its figures are shown from and those figures (its lines alone, for
   * putting in again over the load's own estimates) */
  private fshown = new Map<CityId, { s: Solve; ests: DayDemand[]; round: number }>();
  /** the focus pool's bubble sums, as `pushed` */
  private fpushed = new Map<string, { mask: number; sums: Float32Array; everyday: Float32Array | null }[]>();
  private fversions = new Set<number>();
  /** zones whose pack the focus pool is downloading */
  private ffetching = new Set<CityId>();
  /** ?debug=1: when each focus solve was sent, its zone, and each round shown */
  flog: { version: number; zone: CityId; sentAt: number; openAt?: number; roundAt: number[] }[] = [];
  /** ?debug=1: each pack the focus pool downloaded, when asked and when in */
  fdlog: { id: CityId; at: number; got: number }[] = [];
  /** ?debug=1: when each zone opened in the focus pool */
  fopened: Map<CityId, number> | null = null;
  /** T-2605: the zone the focus pool refines aside (outside a load), or null */
  frefine: CityId | null = null;

  /**
   * T-2605: the zone in view went fine while a whole solve without the barrier runs, which solves it coarse; the
   * main pool's refine of it waits for that solve's final (`firm`), a minute or more on a big game. So a focus
   * pool solves the zone alone at fine level meanwhile, and its figures and bubbles stand for the zone (its lines
   * keep the shown ones) until a solve of the main pool covers it fine. `?perfOff=refineAside` off. T-4220: an edit
   * held behind that solve is the side job's; this waits while a side job is in hand or about to be decided, and refines
   * an edit of the zone in view only once its side job shows nothing (T-4307: a refused side job leaves the wait as it
   * was before T-4220). A side pool kept for a later edit gives way to it.
   */
  private refineAside() {
    const id = activeCity.peek(), s = this.cur, svc = this.latest();
    if (this.lod !== "auto" || this.failed || this.hold || gameLoadArmed() || this.asideJob || this.asideTimer || (this.fpool?.alive && !this.frefine)) return;
    if (!s || !svc || !this.firm() || !this.solveNeed(svc)?.has(id)) return;
    // the zone gone fine, which the solve in hand covers coarse; or an edit of the zone held behind that solve
    const relevel = this.levelPosted.has(id) && this.levelFor(id) === LOD_FINE && s.cities.includes(id) && this.coarseIn(s, id);
    const edit = svc !== s.svc && this.asideNone.has(svc) && this.side?.svc !== svc && (this.edited.get(id) ?? -Infinity) > Date.now() - (performance.now() - s.arrivedAt);
    if (!relevel && !edit) return;
    if (this.fpool?.alive && this.frefine === id && (this.fsolve?.svc ?? svc) === svc) return;
    this.endAside();
    this.endFocus();
    this.startFocus();
    if (!this.fpool) return;
    this.frefine = id;
    this.focusSoon();
  }

  /** T-2605: `s` covers `id` coarse, or not at all */
  private coarseIn(s: Solve, id: CityId): boolean {
    return !s.cities.includes(id) || (s.coarse ?? []).includes(id);
  }

  /** T-2605: the zone refined aside keeps its figures over `s`: `s` covers it coarse, or is of an older network */
  private asideOver(s: Solve): boolean {
    const id = this.frefine, f = id ? this.fsolve ?? this.fshown.get(id)?.s : undefined;
    return !!id && (this.coarseIn(s, id) || (!!f && f.svc !== s.svc && f.version > s.version));
  }

  /** T-2605: `days` with the zone refined aside put back in, over a solve it is kept over */
  private refineOver(days: DayDemand[], s: Solve): DayDemand[] {
    const id = this.frefine, f = id ? this.fshown.get(id) : undefined;
    if (!id || !f || !this.asideOver(s)) return days;
    return this.withZone(days, f.ests, [id], false, new Set(s.layout.map((l) => l.id)), true);
  }

  /** T-2605: `s`'s coarse zones as shown: the zone refined aside is shown fine over it */
  private shownCoarse(s: Solve): CityId[] | undefined {
    const id = this.frefine;
    return id && this.fshown.has(id) && this.asideOver(s) ? (s.coarse ?? []).filter((x) => x !== id) : s.coarse;
  }

  /** T-2605: the focus pool stays past `s`'s final */
  private refineKept(s: Solve): boolean {
    return !!this.fpool?.alive && this.asideOver(s);
  }

  /** A game is loading: start the focus pool (its WASM starts while the save's network is worked out). */
  private startFocus() {
    // (none on a small device: its workers would be a second copy of the zone in the tab's memory, T-2122)
    if (this.fpool || perfOff("focus") || smallDevice() || addressLimited() || this.failed) return;
    // (at most FOCUS_WORKERS solving workers: each worker is memory of its own, T-2045's follow-up)
    const n = Math.max(1, Math.min(this.computeN, Number(new URLSearchParams(location.search).get("focusWorkers")) || FOCUS_WORKERS));
    this.fpool = new FocusPool(n, Array.from({ length: n }, (_, i) => periodsOfWorker(i, n)), {
      message: (i, msg) => this.focusMessage(i, msg),
      opened: (id) => { (this.fopened ??= new Map()).set(id, performance.now()); this.focusSoon(); },
      failed: (why) => { console.warn(`demand: the focus pool stopped (${why})`); this.lastTrouble = `focus pool stopped: ${why}`; this.endFocus(); }, // i18n-ignore
    }, this.viewMsg(), perfOff("focusAux"));
  }

  /** The load's whole solve has landed (or a cached result, a new network, a failure): the zones
   * shown from the focus pool keep their figures until the next publish replaces them. */
  private endFocus() {
    if (!this.fpool && !this.fshown.size) return;
    this.stopFocusPool();
    this.fshown.clear();
    focusZones.value = new Map();
  }

  /** T-2533: a cached result is shown: no zone's figures come from the focus pool, which answers the zone in view's
   * questions until the same network's solve can (`focusAsked`, `retireFocus`) */
  private hideFocus() {
    this.fshown.clear();
    if (focusZones.peek().size) focusZones.value = new Map();
  }

  /** T-2533: with a cached result shown, the zone in view waits for its focus solve: none sent yet, the pool alive and
   * the workers' solve with no round of it; `fsent` is woken when a focus solve is sent or the pool ends */
  private focusComing(): boolean {
    const id = activeCity.peek(), svc = this.latest(), s = this.cur;
    if (!this.fpool?.alive || !this.cacheShown() || this.focusAsked(id) || !svc || !this.solveNeed(svc)?.has(id)) return false;
    const a = s?.ahead?.cities.includes(id) ? s.clusterRes?.get(s.ahead.key)?.published ?? -1 : -1;
    return !(s && s.svc === svc && Math.max(s.published, a) >= 0);
  }
  private fsent: (() => void)[] = [];
  private wakeFocusAsks() {
    for (const r of this.fsent.splice(0)) r();
  }

  /** T-2533: the focus solve asked about `zone` while a cached result of its network is shown */
  private focusAsked(zone: CityId): Solve | null {
    const f = this.fsolve;
    return f && this.fpool?.alive && f.cities[0] === zone && this.cacheShown() && f.svc === this.log.find((s) => s.version === CACHE_VERSION)?.svc ? f : null;
  }

  /** The pool's workers go (their WASM memory with them); questions still waiting on them get no answer. */
  private stopFocusPool() {
    this.fpool?.end();
    this.fpool = this.fsolve = this.frefine = null;
    this.fpushed.clear();
    this.ffetching.clear();
    for (const [id, q] of this.pendingQueries) if (q.focus) { this.pendingQueries.delete(id); q.done([]); }
    this.wakeFocusAsks();
  }

  /**
   * T-2495: the pool's workers end before the load's final once the main pool's own estimates are shown
   * and the pool is no longer ahead of them for any zone it shows (its round no later than theirs) or
   * has settled it (its last round shown): about 0.5 GB for the rest of the load. The zones it settled
   * keep its figures until the final (`withFocus`, `focusZones`); their questions go to the main pool's
   * next round meanwhile (`target`). `?perfOff=focusEnd` keeps the pool to the final.
   */
  private retireFocus() {
    const p = this.fpool, s = this.cur, dv = demandView.peek();
    if (!p?.alive || perfOff("focusEnd") || !s || s.rough || this.frefine) return;
    // T-2533: with a cached result shown, once the same network's solve is sent (the pool and that solve together held
    // 1.0-1.3 GB more on Anita's North America save) and the focus solve's round 0 has answered what was asked of it;
    // at once if that solve already has a round of the zone in view
    const f = this.fsolve;
    if (dv?.version === CACHE_VERSION) {
      if (s.svc !== this.log.find((x) => x.version === CACHE_VERSION)?.svc) return;
      const id = f?.cities[0] ?? activeCity.peek(), a = s.ahead?.cities.includes(id) ? s.clusterRes?.get(s.ahead.key)?.published ?? -1 : -1;
      const asked = f && f.svc === s.svc && (f.published < 0 || [...this.pendingQueries.values()].some((q) => q.focus));
      if (asked && Math.max(s.published, a) < 0) return;
      this.fretiredAt = performance.now();
      return this.stopFocusPool();
    }
    if (!s.showEstimates || s.published < 0 || s.published >= ROUNDS || dv?.version !== s.version) return;
    // (a zone the main pool's estimates do not cover, or a solve that has shown nothing yet, keeps it)
    const done = (id: CityId, f: { s: Solve; round: number }) => f.s.svc !== s.svc || (f.round >= 0 && s.cities.includes(id) && (f.round >= ROUNDS || f.round <= s.published));
    if (f && !done(f.cities[0], { s: f, round: f.published })) return;
    for (const [id, v] of this.fshown) if (!done(id, v)) return;
    this.fretiredAt = performance.now();
    this.stopFocusPool();
  }
  private retireTimer = 0;
  /** ?debug=1: when the pool ended ahead of the load's final (`retireFocus`) */
  fretiredAt: number | null = null;

  /** T-2459: the focus pool has `id` of `svc` in hand: being solved, or shown */
  private focusSolves(svc: Service, id: CityId): boolean {
    return !!this.fpool?.alive && ((this.fsolve?.svc === svc && this.fsolve.cities[0] === id) || this.fshown.get(id)?.s.svc === svc);
  }

  /** The zone in view, reached and with no settled figures, is solved in the focus pool. */
  private focusSoon() {
    const p = this.fpool, id = this.frefine ?? activeCity.peek();
    if (!p?.alive || (!this.frefine && (!viewedZone.peek() || !zonesLoadForView.peek()))) return;
    // T-2533: before the loading game's network, its pack opens there already if the save's stations reach the zone
    const a = this.ahead, early = gameLoadArmed() && !!a && !a.arrived;
    const svc = early ? null : this.latest();
    if (!(svc ? this.solveNeed(svc) : early ? this.reachedSet(a!.svc) : null)?.has(id)) return;
    if (focusZones.peek().get(id) || (this.fsolve?.cities[0] === id && this.fsolve.svc === svc)) return;
    if (!p.holds(id)) {
      if (this.ffetching.has(id)) return;
      this.ffetching.add(id);
      const fl = { id, at: performance.now(), got: 0 };
      this.fdlog.push(fl);
      void this.downloadPack(id).then(([header, bin]) => {
        this.ffetching.delete(id);
        fl.got = performance.now();
        if (this.fpool !== p) return;
        const own = bin.byteOffset === 0 && bin.byteLength === bin.buffer.byteLength ? bin.buffer as ArrayBuffer : bin.slice().buffer;
        p.post({ id, header: new TextDecoder().decode(header), bin: own, level: this.levelFor(id) });
      }, () => this.ffetching.delete(id));
      return;
    }
    if (!svc || !p.open.includes(id)) return; // its `opened` (or the network's arrival) comes back here
    const { net, layout, ids } = packService(svc);
    const version = ++this.version;
    const now = performance.now();
    const s: Solve = { version, svc, layout, results: PERIODS.map(() => null), byRound: Array.from({ length: ROUNDS + 1 }, () => PERIODS.map(() => null)), published: -1, arrivedAt: now, sentAt: now, periodMs: [], setupMs: [], memBytes: [], cities: [id], markets: [], coarse: this.levelFor(id) === LOD_COARSE ? [id] : [], showEstimates: true, rough: true, barrier: true };
    this.fsolve = s;
    this.fversions.add(version);
    this.log.push(s);
    if (this.log.length > 50) this.log.shift();
    this.flog.push({ version, zone: id, sentAt: now, openAt: this.fopened?.get(id), roundAt: [] });
    p.solve({ kind: "solve", version, net, rounds: ROUNDS, barrier: true, first: id, only: [id], keep: true, ids, ...(perfOff("estimate") ? { exact0: true } : {}) });
    this.wakeFocusAsks();
    this.lengthsAhead(); // T-2476: the city view's graph of the zone, of round 0
  }

  private focusMessage(i: number, msg: FromDemand) {
    if (this.reply(i, msg)) return;
    markStep();
    if (msg.kind === "sums") {
      if (!this.fversions.has(msg.version)) return;
      const key = `${msg.version}:${msg.round}:${msg.city}:${msg.day ?? 0}`;
      const list = this.fpushed.get(key) ?? [];
      list.push({ mask: msg.mask, sums: msg.sums, everyday: msg.everyday });
      this.fpushed.set(key, list);
      return;
    }
    const s = this.fsolve;
    if (msg.kind !== "result" || !s || msg.version !== s.version) return;
    if (msg.onlyLines && !s.onlyLines) s.onlyLines = msg.onlyLines;
    const round = msg.summary[0];
    if (msg.part) {
      // T-1543's parts, added up as the main pool's are
      const key = `:${round}:${msg.period}`;
      const m = (s.partsIn ??= new Map());
      const list = m.get(key) ?? [];
      m.set(key, list);
      list[msg.part[0]] = msg;
      if (list.filter(Boolean).length < msg.part[1]) return;
      m.delete(key);
      if (!s.byRound[round]?.[msg.period]) this.focusMessage(i, addParts(list));
      return;
    }
    const results = s.byRound[round];
    if (!results || round <= s.published) return;
    results[msg.period] = msg;
    for (let r = s.published + 1; r <= ROUNDS; r++) {
      if (!s.byRound[r].every(Boolean)) break;
      s.results = s.byRound[r];
      s.published = r;
      // (T-2533: with a cached result shown its round 0 answers the zone's questions; the rounds after would only take
      // the cores from the main pool's solve)
      if (r < ROUNDS && !this.cacheShown()) this.fpool?.go(s.version, r + 1);
      this.publishFocus(s, r);
      if (r > 0) s.byRound[r - 1] = [];
    }
  }

  /** A round of the focus solve: the zone's figures and its lines' shown, alone while nothing else is,
   * else put in the shown result, as a cluster solved ahead is (T-402). */
  private publishFocus(s: Solve, round: number) {
    const id = s.cities[0], final = round >= ROUNDS, res = s.results as Result[];
    if (this.frefine === id) return this.publishRefine(s, round, res);
    const lines = s.onlyLines ? Array.from(s.onlyLines, (l) => s.layout[l]).filter(Boolean) : s.layout;
    this.inOrder(async () => {
      const w = this.cur;
      // T-2533: with a cached result shown its rounds answer questions only (`focusAsked`)
      if (this.cacheShown()) {
        const l = this.flog.find((x) => x.version === s.version);
        if (l) l.roundAt[round] = performance.now();
        return void this.retireFocus();
      }
      // (the load's own solve further on than a round before the last, or landed)
      if (!this.fpool || (!final && w && !w.rough && w.svc === s.svc && w.published > round) || (w && this.cityVersion.get(id) === w.version)) return void this.retireFocus();
      const ests = await this.daysOf(s, lines, round, false, res, true);
      if (!this.fpool) return;
      const shown = this.days[0].view;
      const alone = !shown || (shown.version === s.version);
      const days = alone ? await this.daysOf(s, s.layout, round, false, res) : this.withZone(this.days, ests, [id], false, new Set(s.layout.map((l) => l.id)));
      if (!this.fpool) return;
      // (before the figures go out: the bubbles' effect asks at once where they come from, T-2508)
      this.fshown.set(id, { s, ests, round });
      this.days = days;
      focusZones.value = new Map(focusZones.peek()).set(id, final);
      if (!alone) this.askTurn(); // T-2453
      const l = this.flog.find((x) => x.version === s.version);
      if (l) l.roundAt[round] = performance.now();
      this.lodShown([id], s.coarse);
      this.progress();
      this.checkHold();
      this.retireFocus();
    });
  }

  /** T-2605: a round of the zone refined aside: its own figures and bubbles over the shown ones. Its lines and
   * stations, which carry only its riders here, keep the main pool's, unless it solves an edit held behind the
   * solve in hand (the shown lines are then of the network before). */
  private publishRefine(s: Solve, round: number, res: Result[]) {
    const id = s.cities[0], final = round >= ROUNDS;
    this.inOrder(async () => {
      if (this.frefine !== id || !this.days[0]?.view) return;
      // (the main pool's own solve of the network, sent since, has shown the zone)
      const w = this.cur;
      if (w && w.svc === s.svc && w.version > s.version && (this.cityVersion.get(id) === w.version || w.published >= ROUNDS)) return void this.endFocus();
      // (T-2793: only lines the shown figures have none for: on any other, its stops changed or not, its riders from
      // this zone alone would read far below the shown ones)
      const dv = demandView.peek();
      const lines = (s.onlyLines ? Array.from(s.onlyLines, (l) => s.layout[l]).filter(Boolean) : s.layout).filter((l) => !dv?.lines.has(l.id));
      const days = await this.daysOf(s, lines, round, false, res, true);
      if (this.frefine !== id) return;
      const ests = lines.length ? days : days.map((d) => (d.view ? { ...d, view: { ...d.view, lines: new Map(), stations: new Map(), purposes: new Map() } } : d));
      this.fshown.set(id, { s, ests, round });
      this.days = this.withZone(this.days, ests, [id], false, new Set(s.layout.map((l) => l.id)), true);
      focusZones.value = new Map(focusZones.peek()).set(id, final);
      this.askTurn();
      const l = this.flog.find((x) => x.version === s.version);
      if (l) l.roundAt[round] = performance.now();
      this.lodShown([id], []);
    });
  }

  /** The load's own estimates of a round, with the zones shown from the focus pool put back in where
   * the focus solve is as far on (or settled); T-2605: and the zone refined aside over a solve of it coarse,
   * its final too. */
  private withFocus(days: DayDemand[], s: Solve, round: number, final = false): DayDemand[] {
    // (T-2495: also once the pool's workers have ended, `retireFocus`)
    if (!this.fshown.size) return days;
    days = this.refineOver(days, s);
    if (final) return days;
    const ids = new Set(s.layout.map((l) => l.id));
    for (const [id, f] of this.fshown) {
      if (id === this.frefine) continue;
      if (f.s.svc !== s.svc) continue;
      if (f.round < ROUNDS && f.round < round) {
        this.fshown.delete(id);
        const z = new Map(focusZones.peek());
        z.delete(id);
        focusZones.value = z;
        continue;
      }
      days = this.withZone(days, f.ests, [id], false, ids);
    }
    return days;
  }

  /** `days` with the figures of `ests` put in for `zones` (their views) and the market's if `market`. Lines the
   * network no longer has (`ids`) go. T-2793: `ests` is a solve of these zones alone, so its riders on a line are only
   * theirs (a long line's a few percent of its own): a line goes in only where `days` has none, whatever its stops
   * were solved on (`lineLoadsFor` carries them). Stations likewise when `aside` (a zone refined aside, T-2605); the
   * load's focus zones keep their stations' figures. */
  private withZone(days: DayDemand[], ests: DayDemand[], zones: CityId[], market: boolean, ids: Set<string>, aside = false): DayDemand[] {
    return days.map((d, k) => {
      if (!d.view) return d;
      const est = ests[k];
      const lines = new Map(d.view.lines), stations = new Map(d.view.stations);
      for (const id of lines.keys()) if (!ids.has(id)) lines.delete(id);
      for (const [id, l] of est.view!.lines) if (!lines.has(id)) lines.set(id, l);
      for (const [id, v] of est.view!.stations) if (!aside || !stations.has(id)) stations.set(id, v);
      let purposes = d.view.purposes;
      if (est.view!.purposes) {
        purposes = new Map(purposes);
        for (const [key, p] of est.view!.purposes) {
          const old = purposes.get(key);
          const pl = new Map(old?.lines), ps = new Map(old?.stations);
          for (const [id, l] of p.lines) if (!pl.has(id)) pl.set(id, l);
          for (const [id, v] of p.stations) if (!aside || !ps.has(id)) ps.set(id, v);
          purposes.set(key, { lines: pl, stations: ps });
        }
      }
      const view = { ...d.view, lines, stations, purposes };
      const cities = new Map<CityId, DemandView>();
      for (const [id, v] of d.cities) cities.set(id, { ...v, lines, stations, purposes });
      for (const id of zones) {
        const v = est.cities.get(id);
        if (v) cities.set(id, { ...view, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay, trainShareByPeriod: v.trainShareByPeriod, split: v.split });
      }
      return { view, cities, intercity: market ? est.intercity : d.intercity };
    });
  }

  /** T-4220: the figures on screen as the solving workers hold them (`shown`), until a final replaces them */
  readonly shownSnap = new ShownStore();

  /** T-4220: a solving worker's snapshot of the figures on screen */
  private shownIn(i: number, version: number, snap: Snapshot) {
    if (i >= this.computeN) return;
    const got = this.shownSnap.merge(version, i, snap);
    if (got === "held" && !snap.again) this.asideBroken = false;
    // (a side job waited for the running solve's snapshot from every solving worker)
    if (this.shownSnap.sent(version).size >= this.computeN) this.asideSoon();
    if (!import.meta.env.DEV) return;
    const mb = (b: number) => Math.round(b / 2 ** 20);
    // i18n-ignore
    if (got === "over") return console.info(`T4220 snapshot v${version}: past ${mb(SNAP_MAX_BYTES)} MB with worker ${i}'s, none held until a final`);
    // i18n-ignore
    if (got === "ignored") return console.info(`T4220 snapshot v${version}, worker ${i}: not taken (older than the final on screen, or the figures on screen are not all known)`);
    // i18n-ignore
    if (snap.again) return console.info(`T4220 snapshot v${version} again, worker ${i}: ${snap.pops.length} populations, ${mb(snap.bytes)} MB, ${Math.round(snap.ms)} ms`);
    const sent = [...this.shownSnap.sent(version).values()];
    if (sent.length < this.computeN) return;
    // i18n-ignore
    console.info(`T4220 snapshot v${version}: ${this.shownSnap.pops.size} populations, ${this.shownSnap.periods()} periods, ${mb(this.shownSnap.bytes())} MB, ${Math.round(Math.max(...sent.map((w) => w.ms)))} ms (by worker ${sent.map((w) => Math.round(w.ms)).join(" and ")} ms, ${sent.map((w) => mb(w.bytes)).join(" and ")} MB sent${sent.some((w) => w.shared) ? `, ${(sent.reduce((a, w) => a + w.shared, 0) / 2 ** 20).toFixed(1)} MB of views and network ids already held` : ""})`);
  }

  /** T-4220: a whole solve sends what is on screen where an edit could be solved aside: a focus pool could run here
   * and memory is not short (`?perfOff=editAside`: never) */
  private snapOn(): boolean {
    return !perfOff("editAside") && !perfOff("focus") && !smallDevice() && !addressLimited() && !this.memShort();
  }

  private receive(i: number, msg: FromDemand) {
    this.heard[i] = { at: performance.now(), kind: msg.kind };
    if (msg.kind === "failed") return isMemoryFailure(msg.error) ? this.workerDied(i, msg.error) : this.fail(`worker ${i}: ${msg.error}`); // i18n-ignore
    // i18n-ignore
    if (msg.kind === "dead") return this.workerDied(i, `${msg.error}${msg.panic ? ` (${msg.panic})` : ""}, ${Math.round(msg.memBytes / 2 ** 20)} MB of WASM memory`);
    if (msg.kind === "packOpened") {
      const item = msg.market ? marketItem(msg.id) : msg.id;
      if (i < this.computeN && this.holds(i, item)) this.packAcks.set(item, (this.packAcks.get(item) ?? 0) + 1);
      markStep();
      return this.progressSoon();
    }
    if (msg.kind === "rows") return this.rowsAnswered(msg.id, msg.rows);
    if (this.reply(i, msg)) return;
    markAlive(); // T-950: (not a query's answer, which an idle worker gives while another is stuck)
    markStep();
    if (msg.kind === "unit") { this.inUnit[i] = { version: msg.version, round: msg.round, period: msg.period, at: performance.now() }; return; }
    if (msg.kind === "shown") return this.shownIn(i, msg.version, msg.snap);
    if (msg.kind === "asidePlan" || msg.kind === "asideDone") return; // (a focus pool's alone)
    if (msg.kind === "result") {
      const u = this.inUnit[i];
      if (u) { this.unitMsMax = Math.max(this.unitMsMax, performance.now() - u.at); this.unitTimedIn = u.version; }
      this.inUnit[i] = null;
    }
    if (msg.kind === "loads" && msg.to) {
      // T-3373 B: on to the cluster's other holders
      msg.to.forEach((j, k) => { if (j !== i && j < this.computeN) this.workers[j].postMessage(msg, k === msg.to!.length - 1 ? [msg.loads.buffer] : []); });
      return;
    }
    if (msg.kind === "zones") {
      for (let j = 0; j < this.computeN; j++) if (j !== i) this.workers[j].postMessage(msg);
      return;
    }
    if (msg.kind === "loads") {
      // T-1607: on to the worker solving the period's other side
      const to = msg.side === "cities" ? this.marketsOf.findIndex((qs) => qs.includes(msg.period)) : this.periodsOf.findIndex((qs, j) => j < this.computeN && qs.includes(msg.period));
      if (to >= 0 && to !== i) this.workers[to].postMessage(msg, [msg.loads.buffer]);
      return;
    }
    if (msg.kind === "sums") {
      // (T-2532: an edit's local result's apart, `cellModes` taking them until its whole solve's are all in)
      const key = `${msg.version}:${msg.round}:${msg.city}:${msg.day ?? 0}${msg.local ? ":local" : ""}`;
      const list = this.pushed.get(key);
      if (list) list.push({ mask: msg.mask, sums: msg.sums, everyday: msg.everyday });
      // (T-4265: or the final shown's, as a zone off screen's come after its results, when the next solve went at that final)
      else if (this.cur && (msg.version === this.cur.version || (msg.round >= ROUNDS && msg.version === this.shownOf(msg.city)))) this.pushed.set(key, [{ mask: msg.mask, sums: msg.sums, everyday: msg.everyday }]);
      // (T-4277: a final's, kept through the next solve, added into one as soon as it is whole: those that come after the
      // next solve went would otherwise be held a solving worker's copy each until its end)
      const kept = msg.round >= ROUNDS ? this.pushed.get(key) : undefined;
      if (kept && kept.length > 1 && everyPeriodOnce(kept)) this.pushed.set(key, [mergeSums(kept)]);
      return;
    }
    if (msg.kind === "ready") {
      this.start.workers[i] = { wasmMs: msg.wasmMs, openMs: msg.openMs, warmMs: msg.warmMs, memBytes: msg.memBytes };
      this.noteMem(i, msg.memBytes, msg.heap);
      // T-173: worker 0 has warmed the everyday code up; the held city packs may go
      if (i === 0 && this.held) { const held = this.held; this.held = null; held.forEach((send) => send()); this.postMarkets(); }
      if (!periodsMatch(msg.periods, PERIOD_STRIDE))
        console.error("demand: game/periods.ts disagrees with sim/src/track/params.rs PERIOD_HOURS / PERIOD_ROW", msg.periods);
      if (!this.periodTable.length && i < this.computeN) this.periodTable = msg.periods;
      if (!this.dayTable.length && i < this.computeN) this.dayTable = msg.days;
      if (this.respawn.has(i)) {
        this.workers[i].postMessage(this.viewMsg()); // a replacement's (T-1904; outside any effect)
        return void this.reopen(i);
      }
      for (const id of Object.keys(msg.cityInfo)) this.openedBy(id, i);
      this.isReady.add(i);
      if (++this.ready === this.workers.length) {
        this.status = "ready";

        this.flush();
      }
      this.progress();
      return;
    }
    if (msg.kind === "closed") {
      const w = this.start.workers[i];
      if (w) w.memBytes = msg.memBytes;
      this.noteMem(i, msg.memBytes, msg.heap, false, true);
      this.memSoon();
      return;
    }
    if (msg.kind === "opened") {
      const w = this.start.workers[i];
      if (w) { w.openMs += msg.openMs; w.warmMs += msg.warmMs; w.memBytes = msg.memBytes; }
      this.noteMem(i, msg.memBytes, msg.heap);
      for (const r of msg.rejected ?? []) this.openRejected(r, !!msg.bare);
      const left = this.respawn.get(i)?.left ?? null;
      if (left) {
        for (const id of [...Object.keys(msg.bare ?? msg.cityInfo), ...(msg.rejected ?? []).map((r) => (r.market ? marketItem(r.id) : r.id))]) left.delete(id);
        for (const id of msg.markets) left.delete(marketItem(id));
      }
      if (msg.bare) {
        this.bareOpened(msg.bare);
        if (left && !left.size) this.reopened(i);
        this.progress();
        this.checkHold();
        return;
      }
      for (const [id, box] of Object.entries(msg.boxes ?? {})) this.boxes.set(id, box);
      for (const [id, k] of Object.entries(msg.kinds ?? {})) this.kinds.set(id, k);
      for (const id of Object.keys(msg.cityInfo)) this.openedBy(id, i);
      for (const id of msg.markets) this.openedBy(marketItem(id), i);
      if (left && !left.size) this.reopened(i);
      // T-3373: a big save's extra workers go while its zones open, before each holds them all
      if (i < this.computeN && !this.solving()) this.capPool();
      this.flush();
      this.progress();
      return;
    }
    // T-1008: a worker sends each corridor book once, maybe on a result superseded here
    if (msg.kind === "result" && msg.corridorBook) {
      const t = msg.corridorBook;
      this.corridorBooks.delete(t.key);
      this.corridorBooks.set(t.key, t);
      this.tableAsked.delete(t.key);
      // more than one is in use only while the market changes (`dayCorridors` keeps its own newest)
      if (this.corridorBooks.size > 3) this.corridorBooks.delete(this.corridorBooks.keys().next().value!);
      this.modesBook = t;
      rowsCanCome(t.key);
    }
    const s = this.cur;
    if (!s || msg.version !== s.version) return; // superseded
    if (msg.kind === "done") {
      this.noteMem(i, msg.memBytes, msg.heap, true);
      if (i < this.computeN) this.doneBy.add(i);
      // T-1174: a replacement that has seen a solve through is trusted again
      const r = this.respawn.get(i);
      if (r?.solving) {
        if (r.planned) this.restarted[i] = { wasm: msg.memBytes, need: this.needOf(i) ?? 0 };
        this.respawn.delete(i);
      }
      this.memSoon();
      s.memBytes[i] = msg.memBytes;
      if (msg.clusters) s.clusters = msg.clusters; // ?debug=1's log: clusters kept and solved
      if (msg.periods) (s.periods ??= [])[i] = msg.periods;
      if (msg.local) s.local = msg.local;
      if (i < this.computeN) {
        (s.localHeld ??= [])[i] = !!msg.localHeld;
        if (this.doneBy.size >= this.computeN) this.solveDone(s);
      }
      return;
    }
    if (msg.kind === "work") {
      if (i >= this.computeN) return;
      const w = (s.work ??= []);
      w[i] = { stage: msg.stage, busy: msg.busy, done: msg.done, of: msg.of, unitMs: msg.unitMs, at: performance.now(), left: msg.left, blind: msg.blind, cur: msg.cur, curSize: msg.curSize, sizeMs: msg.sizeMs, blindLate: msg.blindLate, viewBlindLate: msg.viewBlindLate, curLate: msg.curLate, sizeMsLate: msg.sizeMsLate, ran: msg.ran, ranExp: msg.ranExp, ranN: msg.ranN, viewLeft: msg.viewLeft, viewBlind: msg.viewBlind, curView: msg.curView, leftQ: msg.leftQ, ranQ: msg.ranQ, expQ: msg.expQ, nQ: msg.nQ, curQ: msg.curQ };
      if (!msg.busy && msg.done[0] + msg.done[1] > 0) s.lastUnitAt = w[i].at;
      if (!msg.busy && msg.stage === 0 && msg.done[0] > 0) s.localEndAt = w[i].at;
      if (msg.busy && msg.stage === 1 && s.wholeAt === undefined) s.wholeAt = w[i].at;
      if (DEBUG_LOD) (s.workLog ??= []).push({ i, ...w[i] });
      for (let k = 0; k < this.computeN; k++) if (!w[k]) return;
      if (s === this.cur && (!s.refresh || s.refresh.shown)) demandWork.value = w.slice(0, this.computeN);
      // (T-4220: a side job waited for this solve's time left)
      if (this.asideWaitsLeft) { this.asideWaitsLeft = false; this.asideSoon(); }
      return;
    }
    if (msg.kind === "plan") {
      if (i === (s.lead ?? 0)) {
        s.solving = new Set(msg.solving.flat());
        if (s.refresh) s.refresh.solving = msg.solving.flat();
        if (!s.refresh || s.refresh.shown) demandSolving.value = new Set(msg.solving.flat());
        if (msg.ahead) s.ahead = { cities: msg.ahead, key: msg.ahead.join(",") };
        // (T-2459: the focus pool's final of the zone in view may already hold the clock's release)
        if (msg.ahead && s.focused) { this.progress(); this.checkHold(); }
      }
      // T-1733: a worker that solves nothing locally first starts its whole solve at once, so the
      // local result can never be shown whole: the others go on too
      const plans = (s.localPlans ??= []);
      plans[i] = msg.local ?? 0;
      (s.ends ??= [])[i] = !!msg.final;
      const n = this.computeN, all = plans.slice(0, n);
      // T-3616: a network held while the plans came in: now it is known whether it goes at the local result
      if (all.filter((v) => v !== undefined).length === n && this.waiting && s === this.cur) setTimeout(() => { if (this.cur === s) this.flush(); }, 0);
      // (T-4220: a solve with no local stage, its own network solved aside)
      if (all.filter((v) => v !== undefined).length === n) this.asideSoon();
      if (all.filter((v) => v !== undefined).length === n && all.some((v) => v > 0)) {
        if (all.some((v) => !v)) this.wholeGo(s, true);
        else {
          // T-2188: a city edit in every worker: the local results end the solve
          if (s.ends.slice(0, n).every(Boolean) && !perfOff("localEnd")) this.endLocal(s);
          if (s === this.cur && !s.barrier) {
            demandStep.value = { done: 0, of: 2 };
            demandScope.value = 0;
          }
        }
      }
      return;
    }
    if (msg.part) {
      // T-1543: a part of a period's origins: the period goes on once every part is in, added up,
      // unless its worker's own result came first
      s.memBytes[i] = msg.memBytes;
      const key = `${msg.cluster?.cities.join(",") ?? ""}:${msg.summary[0]}:${msg.period}`;
      const m = (s.partsIn ??= new Map());
      const list = m.get(key) ?? [];
      m.set(key, list);
      list[msg.part[0]] = msg;
      if (list.filter(Boolean).length < msg.part[1]) return;
      m.delete(key);
      const round = msg.summary[0];
      const had = msg.cluster ? s.clusterRes?.get(msg.cluster.cities.join(","))?.byRound[round]?.[msg.period] : s.byRound[round]?.[msg.period];
      if (!had) this.receive(i, addParts(list));
      return;
    }
    if (msg.cluster) {
      // T-392: one cluster solved ahead of the rest: a round of it is shown once every period of it
      // is in. Its rounds land in order: with the barrier (T-402) no worker starts one before every
      // worker has the one before; after an edit only the last is sent.
      // (T-1733: a local result keyed apart from its cluster's whole one)
      const key = (msg.cluster.local ? "local:" : "") + msg.cluster.cities.join(",");
      const round = msg.summary[0];
      const m = (s.clusterRes ??= new Map());
      let e = m.get(key);
      if (!e) m.set(key, (e = { byRound: Array.from({ length: ROUNDS + 1 }, () => PERIODS.map(() => null)), published: -1 }));
      const res = e.byRound[round];
      // (T-1543: a worker's own result after the parts were shown changes nothing shown)
      if (!res || round <= e.published) return;
      res[msg.period] = msg;
      if (!res.every(Boolean)) {
        if ((msg.cluster.local || msg.cluster.step) && round >= ROUNDS) this.publishStep(s, e, res, msg.cluster);
        return;
      }
      if (round <= e.published) return;
      e.published = round;
      (e.roundAt ??= [])[round] = performance.now();
      if (round > 0) e.byRound[round - 1] = [];
      if (round < ROUNDS && s.ahead?.key === key) for (const w of this.workers.slice(0, this.computeN)) w.postMessage({ kind: "go", version: s.version, round: round + 1, group: 0 } satisfies ToDemand);
      if (round >= ROUNDS) e.at = performance.now();
      this.publishCluster(s, res as Result[], msg.cluster, round);
      if (this.cacheShown()) this.retireFocus(); // (T-2533)
      this.progress();
      this.checkHold(); // T-421
      return;
    }
    const results = s.byRound[msg.summary[0]];
    // (T-1543: a worker's own result after the parts were shown changes nothing shown)
    if (!results || msg.summary[0] <= s.published) return;
    results[msg.period] = msg;
    (s.roundFirstAt ??= [])[msg.summary[0]] ??= performance.now();
    (s.periodMs[msg.period] ??= [])[msg.summary[0]] = msg.ms;
    if (msg.setupMs) s.setupMs[i] = msg.setupMs;
    s.memBytes[i] = msg.memBytes;
    // Never publish a mixture of population/period rounds: fast workers may be ahead.
    for (let round = s.published + 1; round <= ROUNDS; round++) {
      if (!s.byRound[round].every(Boolean)) break;
      s.results = s.byRound[round];
      s.published = round;
      (s.roundAt ??= [])[round] = performance.now();
      // T-327: every worker has the round, so the next may start
      if (round < ROUNDS) for (const w of this.workers.slice(0, this.computeN)) w.postMessage({ kind: "go", version: s.version, round: round + 1, group: s.ahead ? 1 : 0 } satisfies ToDemand);
      this.publish(s, round);
      if (round === ROUNDS && !s.rough) { this.refreshing = false; memoryResolve.value = false; this.lastStir = performance.now(); this.memSoon(); void this.measurePage(); }
      if (round > 0) s.byRound[round - 1] = []; // release previous typed-array snapshots
    }
    this.progress();
    this.checkHold();
  }

  /** T-2410: `svc`'s stations on no line are `alone` now; whether that changed them (its reach is
   * then worked out again) */
  private setAlone(svc: Service, alone: Service["alone"]): boolean {
    if (JSON.stringify(svc.alone ?? []) === JSON.stringify(alone ?? [])) return false;
    if (alone) svc.alone = alone;
    else delete svc.alone;
    this.reach.delete(svc);
    this.nearZones.delete(svc);
    return true;
  }

  /** A network from the clock worker (T-025), or the test network. */
  submitNetwork(snap: NetworkSnapshot) {
    this.releaseView();
    let svc = serviceFromSnapshot(snap);
    const key = keyOf(svc);
    if (this.pauseHolds(snap, key)) return;
    // the first network after a whole-game load is the one the clock waits for (T-149)
    const gameLoad = takeGameLoadHold();
    if (gameLoad) this.hold = {};
    if (gameLoad) memoryResolve.value = false; // (T-3868: a whole game's load is no solve again)
    if (gameLoad && this.ahead) this.ahead.arrived = true;
    if (gameLoad) this.cacheSolve = null;
    // T-1539: the network the loaded save works out, for the cache written for it; and the one a
    // cached result shown ahead of it was for: the same network is that very service
    const early = this.early;
    this.early = null;
    if (gameLoad && this.loadKey && !this.loadKey.svcKey) Object.assign(this.loadKey, { svcKey: key, snapshot: snap });
    else if (gameLoad) this.loadKey = null;
    if (early) this.aheadLog = { at: performance.now(), same: early.key === key, shown: !!early.shown };
    if (early?.key === key) {
      this.setAlone(early.svc, svc.alone);
      svc = early.svc;
    } else if (early?.shown) this.dropEarly(early.svc);
    // T-2658: the world as it is now is in this network (a world's state comes before the snapshot made after it)
    const wv = world.peek()?.version ?? -1;
    networkWorld.value = wv;
    // T-2849: demand that failed will solve nothing, so no edit waits on it
    if (this.failed) { finalWorld.value = wv; return; }
    if (key === this.lastKey) {
      const same = this.latest();
      if (same) this.netWorld.set(same, wv);
      if (same && !this.waiting && this.cur?.svc === same && demandView.peek()?.final && demandView.peek()?.version === this.cur.version) finalWorld.value = wv;
      // nothing demand reads changed: the solve in hand (or waiting) is the loaded game's; stations
      // built or removed on no line may reach other zones (T-2410)
      const held = this.latest();
      // (a game load of the network in hand: also what opened ahead of it, which `flush` left while it was on its way)
      if ((held && held !== svc && this.setAlone(held, svc.alone)) || gameLoad) this.ensure();
      if (this.hold && !this.waiting) this.hold.version = this.cur?.version;
      this.checkHold();
      return;
    }
    this.lastKey = key;
    this.netWorld.set(svc, wv);
    this.netSeq.set(svc, ++this.netCount); // (T-4220)
    // T-2560: an edit after the cache's card went is an edit like any other
    this.cacheSolve = null;
    this.lastStir = performance.now();
    if (!gameLoad) this.editAt = this.lastStir;
    if (gameLoad) this.uncovered = [];
    else this.uncovered.push({ svc, minute: clock.minute.peek() });
    if (this.uncovered.length > 50) this.uncovered.splice(1, 1); // (the oldest's minute is what counts)
    demandBehindFrom.value = this.uncovered[0]?.minute ?? null;
    // T-2045: the focus pool is a game load's; an edit's network ends it (T-4220: the side pool stays for the edit)
    if (!gameLoad) this.endFocus();
    else { this.endAside(); this.dropSide(); if (svc.stations.length) this.startFocus(); }
    if (gameLoad) { this.shownSnap.clear(this.version); this.pushed.clear(); } // (T-4220, T-4265: another game's figures)
    demandUpdating.value = true;
    demandStep.value = { done: 0, of: 1 }; // T-1790: no count until the solve's own steps are known
    // T-1223: a game loaded with no stations ahead makes its fine set afresh; an edit marks its zones
    const prev = this.latest();
    if (gameLoad && !this.ahead) this.lodReset(null);
    else if (!gameLoad && prev) this.noteEdits(prev, svc);
    this.lodSoon();
    // T-840: the game's own load replaces what loaded before its network was known; a game loaded while
    // the last one's bar is up starts the bar over (T-884: not the pick's own wait, which goes on)
    if (gameLoad && this.job) {
      const same = this.job.wait === gameWait().seq;
      this.job = { ...this.newJob(startup.peek() === "loading"), ...(same ? { best: this.job.best, anchor: this.job.anchor, since: this.job.since, shares: this.job.shares, viaCache: this.job.viaCache, cacheReadyAt: this.job.cacheReadyAt } : {}) };
    }
    this.send(svc, performance.now());
    this.focusSoon();
    // T-328: a whole game's network: its last final result may be kept beside the save (shown
    // already if it was found ahead of the network, T-1539)
    if (this.hold && this.cacheOn && !(early?.shown && svc === early.svc)) void this.tryCache(svc);
    else if (this.hold && this.cacheShown()) this.checkHold(); // (T-2560: shown ahead of the network)
  }

  // --- T-3427: edits made while paused, solved once the game runs ---

  /** the newest network held while paused, or null */
  private pausedSnap: NetworkSnapshot | null = null;
  private pausedAt = 0;
  private pauseWatch: (() => void) | null = null;

  /** Whether `snap` waits for the game to run ("Recalculate trips while paused" off and the game paused).
   * A game's own load never waits, nor a network demand reads the same as the one in hand. */
  private pauseHolds(snap: NetworkSnapshot, key: string): boolean {
    const waits = !display.peek().recalcWhilePaused && !clock.running.peek();
    if (!waits || gameLoadArmed() || this.failed || key === this.lastKey) {
      this.pausedSnap = null;
      demandHeld.value = false;
      return false;
    }
    if (!this.pausedSnap) this.pausedAt = performance.now();
    this.pausedSnap = snap;
    demandHeld.value = true;
    this.pauseWatch ??= effect(() => {
      if (!display.value.recalcWhilePaused && !clock.running.value) return;
      const s = this.pausedSnap;
      this.pausedSnap = null;
      demandHeld.value = false;
      // (outside the effect, so the solve's own signal reads are not tracked here)
      if (s) queueMicrotask(() => { if (!gameLoadArmed()) this.submitNetwork(s); });
    });
    return true;
  }

  // --- T-1539: a loading save's cached result, shown before its network arrives ---

  /** the loading save's network key, and the network its load sent (for the cache written for it) */
  private loadKey: { key: string; svcKey?: string; snapshot?: NetworkSnapshot } | null = null;
  /** the network of the cached result found for the loading save, and whether it is shown */
  private early: { svc: Service; key: string; shown?: boolean } | null = null;
  /** the save key of the cached result shown, if it held one */
  private cachedAhead: string | null = null;
  /** the network's arrival after a cached result found ahead of it: when, and whether it was the same */
  aheadLog: { at: number; same: boolean; shown: boolean } | null = null;

  /**
   * A save is loading whose network inputs have key `key` (demandCache.ts `saveNetKey`). A cached
   * result written for the network a load of the same inputs worked out holds that network, so it
   * shows now, through the same checks as at the network's arrival (`tryCache`), while the clock
   * worker works the network out again. The network that arrives is compared with it.
   */
  netAhead(key: string) {
    this.loadKey = { key };
    this.early = null;
    if (!this.cacheOn || perfOff("cacheAhead") || this.failed) return;
    void (async () => {
      const stored = await readDemandAhead<StoredDemand>(key);
      if (!stored) this.cacheLogs.push({ at: performance.now(), ahead: true, miss: "none", hash: key });
      if (!stored?.ahead || stored.ahead.save !== key || this.loadKey?.key !== key || !gameLoadArmed()) return;
      const svc = serviceFromSnapshot(stored.ahead.snapshot);
      this.early = { svc, key: keyOf(svc) };
      await this.tryCache(svc, stored);
    })();
  }

  /** The network arrived and is not the one the cached result shown ahead of it was for. */
  private dropEarly(svc: Service) {
    const i = this.log.findIndex((s) => s.version === CACHE_VERSION && s.svc === svc);
    if (i >= 0) this.log.splice(i, 1);
    this.expectCache(null, false);
    if (!this.cacheShown()) return;
    queueMicrotask(() => this.wakeFocusAsks());
    this.cachedCities = new Set();
    this.cachedHash = this.cachedAhead = null;
    this.cachedModes = null;
    this.cachedStandIn = false;
    this.cachedLocal = false;
    this.days = DEMAND_DAYS.map(() => ({ view: null, cities: new Map(), intercity: null }));
  }

  /** Solve the current network again under a new version (timing runs). */
  resubmit() {
    if (this.cur) this.send(this.cur.svc, performance.now());
  }

  /** Solve `svc` once every city it needs is open (T-149); until then it waits, and the last
   * published result stays. */
  private send(svc: Service, arrivedAt: number) {
    this.waiting = { svc, arrivedAt };
    this.ensure();
  }

  private dispatch(svc: Service, arrivedAt: number, refresh = false) {
    this.capPool();
    this.doneBy.clear();
    clearTimeout(this.refreshTimer);
    if (this.freshParts) this.refreshing = true;
    const { net, layout, ids } = packService(svc);
    const version = ++this.version;
    if (this.hold) this.hold.version = version;
    const cities = CITIES.map((c) => c.id).filter((id) => this.opened.has(id));
    // the load in hand ends with this solve if it covers every city of it the solves must (T-325:
    // a zone opened bare needs none)
    const need = this.solveNeed(svc);
    if (this.job && this.jobNeedsSolve(this.job) && this.job.cities.every((id) => cities.includes(id) || (need !== null && !need.has(id)))) this.job.version = version;
    this.keepShownSums(null);
    // T-232: a re-solve after an edit keeps the last final result shown until its own final round,
    // so the figures never step through the rounds; estimates show only where nothing final is
    // shown yet: before the first final result or after a whole-game load. T-2793: a zone just opened
    // (`newZones`) takes the rounds' figures for itself alone, and every line keeps its final figures.
    const shown = cityDemandViews.peek();
    const unshown = cities.some((id) => !shown.has(id));
    const showEstimates = !demandView.peek()?.final || (!!this.hold && !this.cacheShown());
    const newZones = !showEstimates && unshown;
    // (T-2560: a cached result's network solved again after its card went, as under the card)
    // (T-3373 B: under `?split=1` every solve is whole, round by round with every holder: stage 6 adds the edits' own)
    const barrier = showEstimates || newZones || !!this.hold || svc === this.cacheSolve || !!this.holding;
    const markets = [...this.opened].filter(isMarket).map(marketZone);
    // T-3616: an edit's solve sent before the one in hand started its whole solve follows on from it
    const prev = this.cur;
    // (T-2188: an edit stopping the refresh begins its own)
    const chainFrom = !barrier && prev && prev.barrier === false && !prev.rough && !prev.refresh && !prev.wholeSent && prev.published < ROUNDS ? prev.chainFrom ?? prev.sentAt : undefined;
    // T-3925: and one sent before the one in hand's final, whose whole solve it stops (`restarts`)
    const unfinished = !barrier && prev && prev.barrier === false && !prev.rough && !prev.refresh && prev.published < ROUNDS;
    const restartFrom = unfinished ? prev.restartFrom ?? prev.chainFrom ?? prev.sentAt : undefined;
    if (unfinished && this.wholeRuns(prev)) {
      prev.cutAfter = Math.round(performance.now() - prev.sentAt);
      solveRestarted.value++;
    }
    this.cur = { ...(chainFrom !== undefined ? { chainFrom } : {}), ...(restartFrom !== undefined ? { restartFrom } : {}), version, svc, layout, results: PERIODS.map(() => null), byRound: Array.from({length: ROUNDS + 1}, () => PERIODS.map(() => null)), published: -1, arrivedAt, sentAt: performance.now(), periodMs: [], setupMs: [], memBytes: [], cities, markets, coarse: this.coarseOf(cities), sent: { cities: [...cities], markets: [...markets] }, showEstimates, newZones, barrier, ...(refresh ? { refresh: { shown: false } } : {}) };
    this.log.push(this.cur);
    if (this.log.length > 50) this.log.shift();
    if (this.cacheShown()) this.retireFocus(); // (T-2533)
    // (T-2188: the refresh shows nothing unless an edit waits for it, `showRefresh`)
    if (!refresh) {
      demandUpdating.value = true;
      // T-1790: after an edit the rounds all land at the end, so the steps are the local solve and the
      // whole one (`plan` says whether there is a local one), not rounds
      demandStep.value = { done: 0, of: barrier ? ROUNDS + 1 : 1 };
      demandWork.value = null;
      editHeld.value = null;
      demandScope.value = cities.length;
      // T-327: rounds in step only where clicks are answered by round (`target`): it costs the final
      // about 0.7 s on the real save
      demandSolving.value = null;
    }
    const next = this.firstZones().slice(1);
    // T-2624: after an edit the free-flow preview only gives new lines and lines with other stops their first
    // figures (`publishNewLines`); with every line as the shown figures' solve had it, its parts are work for nothing
    // (8-9 s a worker on the New York save)
    const base = this.log.find((x) => x.version === demandView.peek()?.version);
    const had = base && new Map(base.layout.map((l) => [l.id, l.stops.join(",")]));
    const noPreview = !barrier && !!had && !perfOff("preview") && layout.every((l) => had.get(l.id) === l.stops.join(","));
    // T-2459: the focus pool solves the zone in view to its last round, so the city view's zone's
    // cluster goes ahead of it
    const focused = barrier && next.length > 0 && !perfOff("focusAhead") && this.focusSolves(svc, activeCity.peek());
    if (focused) this.cur.focused = true;
    const msg: ToDemand = { kind: "solve", version, net, ids, rounds: ROUNDS, barrier, ...(barrier ? { later: loadPace().later } : {}), first: activeCity.peek(), ...(next.length ? { next } : {}), ...(focused ? { focused } : {}), ...(perfOff("clusters") ? { whole: true } : {}), ...(perfOff("split") ? { oneMarket: true } : {}), ...(perfOff("reach") ? { wide: true } : {}), ...(perfOff("chunks") ? { exactMarket: true } : {}), ...(perfOff("farChunks") ? { nearChunks: true } : {}), ...(perfOff("periods") ? { fresh: true } : {}), ...(perfOff("estimate") ? { exact0: true } : {}), ...(this.freshParts ? { freshParts: true } : {}), ...(perfOff("noise") ? { exactKeep: true } : {}), ...(perfOff("steps") ? { oneSend: true } : {}), ...(perfOff("wholeSteps") ? { roundFirst: true } : {}), ...(noPreview ? { noPreview } : {}),...(LOCAL_SHARE ? { localShare: LOCAL_SHARE } : {}), ...(refresh ? { refresh } : {}) };
    if (!barrier && this.snapOn()) msg.snap = this.cur.snap = true;
    this.freshParts = false;
    this.postSolve(msg);
    if (barrier) this.askTurn();
    this.progress(); // T-243: a held game's solve shows its progress even with nothing to load
    this.releaseRespawnWaits();
  }

  /** A solve to every worker, each solving worker told its part of the origins (T-1543). */
  private postSolve(msg: Extract<ToDemand, { kind: "solve" }>) {
    const n = this.computeN;
    // T-3373 B: under `?split=1` each solving worker its holding plan (no T-1543 parts: every solve is whole)
    const plan = this.holding && this.splitPlan(msg.version);
    if (plan) {
      // (a rough solve: its zone's holder alone solves it, and says what it solves; the other solving workers leave
      // what they were solving and wait, never ready for it, until the whole solve supersedes it)
      const s = this.log.find((x) => x.version === msg.version);
      if (s) s.plan = plan;
      if (s && msg.only) s.lead = plan.active[0];
      return this.workers.forEach((w, i) => w.postMessage(i < n ? { ...msg, split: { ...plan, me: i } } : msg));
    }
    this.workers.forEach((w, i) => w.postMessage(n > 1 && i < n ? { ...msg, part: [i, n] } : msg));
  }
  /** T-3373 B: the solve's holding plan: its zones (the solve's, in the registry's order) with their boxes and holders */
  private splitPlan(version: number): Omit<SplitPlan, "me"> | null {
    const s = this.log.find((x) => x.version === version);
    if (!s || !this.holding) return null;
    const cities = s.cities.map((id) => ({ id, box: this.boxes.get(id) ?? null, by: this.solvingHolders(id)[0], kinds: this.kinds.get(id) ?? 0 }));
    const market = s.markets.length ? this.solvingHolders(MARKET)[0] : null;
    const active = [...new Set([...cities.map((c) => c.by), ...(market === null ? [] : [market])])].sort((a, b) => a - b);
    // (a solve covering nothing still gives every period its figures: worker 0's)
    return { cities, market, active: active.length ? active : [0] };
  }

  // --- Demand views (T-078; types and meaning in game/demand.ts) ---------------------------------

  /** T-2045: a question to the focus pool's aux worker */
  private askFocusAux(make: (id: number) => Query, zone?: CityId): Promise<Reply[]> {
    return this.ask(make, [this.computeN], [], zone, undefined, true);
  }

  /** Send `make(id)` to the given workers (all solving workers by default) and collect their replies. */
  private ask(make: (id: number) => Query, only?: number[], transfer: Transferable[] = [], zone?: CityId, t?: Target, toFocus = false): Promise<Reply[]> {
    let which = only ?? this.workers.slice(0, this.computeN).map((_, i) => i);
    this.syncFocus();
    const id = this.nextQuery++;
    // `zone`: a nearby zone's question (T-276), about that zone whatever the focus does meanwhile
    const city = zone ?? activeCity.peek(), epoch = this.focusEpoch;
    const q = make(id);
    // T-2045: a question about a focus solve goes to the focus pool, the same worker index (the aux's
    // to its aux, any other to its solving workers)
    const fp = this.fpool, version = (q as { version?: number }).version;
    const pool = fp?.alive && (toFocus || (version !== undefined && this.fversions.has(version))) ? fp : null;
    // (one solving worker asked by name stays one: `flows`, its sums transferred, went to both of two focus
    // workers, the second post threw on the detached buffer and the far end never came, T-2531)
    if (pool) which = which.includes(this.computeN) ? [pool.auxAt] : only?.length === 1 ? [Math.min(only[0], pool.n - 1)] : pool.workers.slice(0, pool.n).map((_, i) => i);
    // T-3373 B: about a split solve, to the workers holding what it asks about
    const route = pool ? null : this.splitRoute(q, city, only, t);
    if (route) {
      if (!route.to.length) return Promise.resolve([]);
      which = route.to;
    }
    const workers = pool?.workers ?? this.workers;
    return new Promise((done) => {
      this.pendingQueries.set(id, { left: which.length, replies: [], done: replies => done(zone !== undefined || epoch === this.focusEpoch ? (route ? covered(replies, route.need) : replies) : []), focus: !!pool });
      this.qlog.push({ id, kind: q.kind, sent: performance.timeOrigin + performance.now(), replies: [] });
      if (this.qlog.length > 200) this.qlog.shift();
      // T-260: the kind of day shown; each answer weighs its periods by it
      const round = t?.round ?? demandView.peek()?.round ?? -1;
      for (const i of which) workers[i].postMessage({...q, city, round, day: shownDay.peek(), ...(t?.wait ? { wait: true } : {})}, transfer);
    });
  }

  /**
   * T-3373 B (`?split=1`): the workers a question about a split solve goes to, by its plan (demandProtocol
   * `ZONE_QUERIES`), and the populations their answers must cover between them; null as without the split (no
   * split, the aux worker named, or a question of another kind). Each zone and the market have one solving holder.
   */
  private splitRoute(q: Query, city: CityId, only: number[] | undefined, t?: Target): { to: number[]; need: string[] | null } | null {
    if (!this.holding || only?.includes(this.computeN)) return null;
    const zoneQ = ZONE_QUERIES.has(q.kind), marketQ = MARKET_QUERIES.has(q.kind);
    if (!zoneQ && !marketQ && q.kind !== "transfers") return null;
    const version = (q as { version?: number }).version;
    const plan = (t?.solve?.version === version ? t?.solve : this.log.find((x) => x.version === version))?.plan;
    // (a question about a solve with no plan here, a cached result's, goes nowhere and is not answered)
    if (!plan) return { to: [], need: [] };
    const n = this.computeN, at = (w: number | null | undefined) => (w !== null && w !== undefined && w < n ? [w] : []);
    if (zoneQ) return { to: at(plan.cities.find((c) => c.id === city)?.by), need: [city] };
    if (marketQ) return { to: at(plan.market), need: [MARKET_POP] };
    return { to: plan.active.filter((w) => w < n), need: null };
  }

  /**
   * T-327: where a view's question goes. While the running solve's estimates are shown (startup, a
   * zone just opened), or a cached result of the same network is, the workers are asked about the
   * newest round any of them is in, and each answers on reaching it: one round ahead of the shown
   * figures at most. Otherwise the shown result's version and round, as a worker holds them only
   * when idle. Not after an edit: the workers then hold a network the map does not show yet.
   */
  private target(zone: CityId = activeCity.peek(), anyShown = false): Target | null {
    const dv = demandView.peek();
    // T-2045: a zone shown from the focus pool is asked about there, as a solve stepping through rounds
    const f = this.fpool?.alive ? this.fshown.get(zone) : undefined;
    if (f) {
      const view = dv?.version ?? f.s.version;
      return f.round >= ROUNDS ? { version: f.s.version, round: ROUNDS, view, wait: false, solve: f.s } : { version: f.s.version, round: f.round + 1, view, wait: true, solve: f.s, shown: f.round };
    }
    // T-2533: with a cached result shown, the zone in view solved alone in the focus pool, which holds its round 0
    // (or the round it was in when the cache showed: no further `go`)
    const fa = this.focusAsked(zone);
    if (fa) return { version: fa.version, round: Math.max(0, fa.published), view: dv?.version ?? CACHE_VERSION, wait: true, solve: fa, shown: fa.published };
    const s = this.cur;
    // T-402: the zone in view's cluster solved ahead counts its own rounds (the workers tell which
    // rounds a question means by its zone)
    const ahead = s?.ahead?.cities.includes(zone) ? s.clusterRes?.get(s.ahead.key)?.published ?? -1 : null;
    const published = ahead ?? s?.published ?? -1;
    // nothing shown yet (startup, no cache): asked of round 0, whose answer is in hand when it shows
    if (!dv) return s && published < 0 ? { version: s.version, round: 0, view: s.version, wait: true, solve: s, shown: -1 } : null;
    const cv = this.cityVersion.get(zone);
    if (s && cv === s.version) return { version: cv, round: ROUNDS, view: dv.version, wait: false, solve: s };
    // T-1205: after a worker change the shown result is not in every worker: asked of the new final
    if (s && this.refreshing && dv.version !== s.version && published < ROUNDS) return { version: s.version, round: ROUNDS, view: dv.version, wait: true, solve: s, shown: -1 };
    const cached = dv.version === CACHE_VERSION && this.log[0]?.version === CACHE_VERSION && this.log[0].svc === s?.svc;
    // T-2453 (`anyShown`): or a solve with the barrier, for a zone it covers that shows no settled
    // figures, whatever the view is of (during a game load, the game before's final with the load's
    // estimates and the focus pool's figures put in)
    const unsettled = anyShown && !perfOff("selected") && !!s?.barrier && !s.rough && s.cities.includes(zone) && this.unsettledZone(zone);
    if (s && published < ROUNDS && (dv.version === s.version || cached || unsettled)) {
      // no worker starts a round before every worker has the one before (`go`), so every worker
      // has the round after the published one, or will
      return { version: s.version, round: published + 1, view: dv.version, wait: true, solve: s, shown: published };
    }
    // T-392: a zone shown from a cluster solved ahead of its solve's final is asked about there
    if (cv !== undefined) return { version: cv, round: ROUNDS, view: dv.version, wait: false, solve: this.log.find((x) => x.version === cv) };
    return { version: dv.version, round: dv.round, view: dv.version, wait: false, solve: this.log.find((x) => x.version === dv.version) };
  }

  /** The answer asked for at `t` still goes with what is shown. */
  private still(t: Target): boolean {
    const v = demandView.peek()?.version;
    if (this.fversions.has(t.version)) return (this.fshown.get(activeCity.peek())?.s ?? this.focusAsked(activeCity.peek()))?.version === t.version;
    return t.wait ? this.cur?.version === t.version && (v === undefined || v === t.view || v === t.version) : v === t.view;
  }

  /**
   * T-327: a view's question (`key`, with the zone and day added) at the target. While the workers
   * are ahead of the shown round, an answer already asked for the shown round is given (the view
   * asks again at each publish, and that answer came in just before it), and the next round's is
   * asked for at once, so each publish finds its answer waiting. A worker already past the round
   * asked about answers nothing; the question then goes to the newer round.
   */
  private viaRound<T>(key: string, run: (t: Target) => Promise<T | null>, tries = 2): Promise<T | null> {
    // T-2533: the zone in view of a cached result, which the focus pool is about to solve: asked of that solve
    if (this.focusComing()) return new Promise<void>((r) => this.fsent.push(r)).then(() => this.viaRound(key, run, tries));
    const t = this.target();
    // T-3492: a worker being replaced holds nothing yet: asked once the pool's solve again is out (`target` then
    // waits for its final); not a question for the focus pool, whose workers were not replaced
    if (this.reopening() && !(t && this.fversions.has(t.version))) return new Promise<void>((r) => this.respawnWaits.push(r)).then(() => this.viaRound(key, run, tries));
    if (!t) return Promise.resolve(null);
    if (!t.wait) return run(t);
    const dv = demandView.peek();
    const shown = t.shown ?? (dv?.version === t.version ? dv.round : -1);
    // (T-402: rounds are counted per zone while the zone in view's cluster is solved ahead)
    const zone = `${activeCity.peek()}|`;
    for (const [k, e] of this.aheadP) if (e.version !== t.version || (k.startsWith(zone) && e.round < shown)) this.aheadP.delete(k);
    const base = `${zone}${shownDay.peek()}|${key}|${t.version}|`;
    const at = (round: number) => {
      let e = this.aheadP.get(base + round);
      if (!e) {
        const p = run({ ...t, round });
        this.aheadP.set(base + round, (e = { version: t.version, round, p }));
        p.then((v) => { if (!v && this.aheadP.get(base + round)?.p === p) this.aheadP.delete(base + round); });
      }
      return e.p as Promise<T | null>;
    };
    const next = at(t.round);
    const p = shown >= 0 && this.aheadP.has(base + shown) ? at(shown) : next;
    return p.then((v) => {
      if (v || tries <= 0) return v;
      const t2 = this.target();
      return t2?.wait && t2.version === t.version && t2.round > t.round ? this.viaRound(key, run, tries - 1) : null;
    });
  }

  cityCells(zone?: CityId): Promise<CityCells | null> {
    this.syncFocus();
    const other = zone !== undefined && zone !== activeCity.peek() ? zone : undefined;
    // T-2045: a zone the focus pool holds and the main pool has not opened yet: its cells from there
    const viaFocus = (id: CityId) => !this.opened.has(id) && !!this.fpool?.open.includes(id);
    // (T-4265: a zone opening again at another level is held by the aux worker as before meanwhile, which a solving worker's
    // solve in hand may hold up for minutes)
    const held = (id: CityId) => viaFocus(id) || (this.releveling.has(id) && !this.auxShared);
    const load = (id: CityId) => (held(id) ? Promise.resolve(true) : this.whenOpen(id)).then(async (ok): Promise<CityCells | null> => {
      if (!ok) return null;
      const [r] = viaFocus(id) ? await this.askFocusAux((q) => ({ kind: "cells", id: q, frame: frameOf() }), other) : await this.ask((q) => ({ kind: "cells", id: q, frame: frameOf() }), [this.auxAt], [], other);
      if (r?.kind !== "cells" || r.xy.length === 0) return null;
      return { n: r.xy.length / 2, xy: r.xy, h3: r.h3, zone: r.zone, zoneM: r.zoneXY[0], zoneXY: r.zoneXY.subarray(1) };
    });
    if (other !== undefined) {
      // a nearby zone's (T-276): kept per zone and world frame; callers share the arrays and only read them
      const key = `${other}:${worldFrameEpoch.peek()}`;
      let p = this.otherCells.get(key);
      if (!p) {
        this.dropCells(other);
        this.otherCells.set(key, (p = load(other)));
        p.then((v) => { if (!v && this.otherCells.get(key) === p) this.otherCells.delete(key); });
      }
      return p;
    }
    // T-2756: a zone's cells from an earlier visit (they change only with its pack and the frame), so a return needs
    // no worker: after a memory restart the workers answer only once the whole game is solved again
    const focus = activeCity.peek(), frame = worldFrameEpoch.peek(), key = `${focus}:${frame}`;
    const kept = this.otherCells.get(key);
    if (kept) return kept;
    // waits for the viewed city to open (T-149: it may be loading after a pan); a failed answer is
    // not kept, so the bubbles' next ask tries again (T-1493)
    if (!this.cellsP) {
      const p = (this.cellsP = load(focus));
      p.then((v) => {
        if (!v) { if (this.cellsP === p) this.cellsP = null; }
        else if (this.opened.has(focus) && worldFrameEpoch.peek() === frame) { this.dropCells(focus); this.otherCells.set(key, p); }
      });
    }
    return this.cellsP;
  }
  /** Nearby zones' cells (T-276) and the focus's from earlier visits (T-2756), `zone:frame epoch`. */
  private otherCells = new Map<string, Promise<CityCells | null>>();
  /** A zone's cells in every frame, when it closes or its frame is out of date. */
  private dropCells(id: string) {
    for (const k of this.otherCells.keys()) if (k.slice(0, k.lastIndexOf(":")) === id) this.otherCells.delete(k);
    this.zsq.delete(id as CityId);
  }

  /** One purpose's local trip legs per cell (T-158's filter): work from the commute sums alone,
   * an everyday purpose from the workers' everyday layers, longer trips none. Cached per solve,
   * round and purpose, like `cellModes`. */
  purposeModes(purpose: PurposeFilter, zone?: CityId): Promise<CellModes | null> {
    // `zone`: a nearby zone's legs (T-276), not cached here (map/commuterView.ts keeps its build)
    const other = zone !== undefined && zone !== activeCity.peek() ? zone : undefined;
    if (purpose === "all") return this.cellModes(true, other);
    this.syncFocus();
    const version = demandView.peek()?.version;
    if (version === undefined) return Promise.resolve(null);
    const round = demandView.peek()?.round ?? 0;
    const day = shownDay.peek();
    const src = this.cityVersion.get(other ?? activeCity.peek()); // T-392, as in cellModes
    const at = src !== undefined ? shownTarget(src, ROUNDS, version) : shownTarget(version, round);
    const key = `${version}:${round}:${day}:${purpose}:${src ?? ''}`;
    const cached = other === undefined ? this.purposeP.get(key) : undefined;
    if (cached) return cached;
    const p = (async (): Promise<CellModes | null> => {
      const city = other ?? activeCity.peek();
      if (purpose === "work") return this.cellModes(false, other);
      const cells = await this.cityCells(other);
      if (!cells) return null;
      // a shown purpose may count more than one of the model's (school and university, T-192)
      const kinds = membersOf(purpose).map((m) => EVERYDAY_KEYS.indexOf(m)).filter((k) => k >= 0);
      let modes = new Float32Array(8 * cells.n);
      const bare = this.isBare(city);
      for (const kind of kinds) {
        if (bare) {
          // T-325: the aux worker's no-rail pass, one part per solving worker's periods
          const [r] = await this.ask((id) => ({ kind: "bare", id, version, purpose: kind }), [this.auxAt], [], other);
          if (r?.kind !== "bare" || !r.commute) return null;
          for (const p of bareParts(r)) if (p.length === modes.length) addInto(modes, p);
          continue;
        }
        let mask = 0;
        for (const r of byMask(await this.ask((id) => ({ kind: "purposeModes", id, version: at.version, purpose: kind }), undefined, [], other, other === undefined ? this.liveAt(at) : at))) {
          if (r.kind !== "purposeModes" || !r.modes) return null;
          mask |= r.mask;
          if (r.modes.length === modes.length) addInto(modes, r.modes);
        }
        if (mask !== ALL_PERIODS) return null;
      }
      // T-3503: the purpose's trips with other places (airports' at their terminals, by the local legs)
      const tg = other === undefined ? this.liveAt(at) : at, local = kinds.includes(AIRPORT_KIND) ? modes.slice() : null;
      await this.regionalInto(modes, city, at.version, other, tg, kinds.reduce((b, k) => b | (1 << (2 + k)), 0));
      if (local) await this.airportAcross(modes, local, cells, city, at.version, other, tg);
      if (!stillShown(other, city, version, day)) return null;
      return cellModesOf(version, day, modes);
    })();
    if (other !== undefined) return p;
    // one solve's purposes at a time; a failed answer is not kept
    for (const k of this.purposeP.keys()) if (!k.startsWith(`${version}:${round}:`)) this.purposeP.delete(k);
    this.purposeP.set(key, p);
    p.then((v) => { if (!v && this.purposeP.get(key) === p) this.purposeP.delete(key); });
    return p;
  }

  /** T-3500: the airport trips from across the zone line added to `modes` (eight blocks) at a zone's
   * airports: each 6 km zone's spread over its terminal cells by their airport legs in `weights`
   * (eight blocks; T-3503: the residents' end comes with the other trips, `regionalInto`). As they
   * were when the workers cannot say. */
  private async airportAcross(modes: Float32Array, weights: Float32Array, cells: CityCells, city: CityId, version: number, other: CityId | undefined, t: Target, keep?: string): Promise<void> {
    const n = cells.n;
    if (modes.length !== 8 * n || weights.length !== 8 * n) return;
    const ends = [[], [3, 4, 5, 7]].map((blocks) => {
      const at: number[] = [], w: number[] = [];
      for (let c = 0; c < n && blocks.length; c++) {
        let v = 0;
        for (const b of blocks) v += weights[b * n + c];
        if (v > 0) at.push(c), w.push(v);
      }
      const xy = new Float32Array(2 * at.length);
      at.forEach((c, i) => {
        // the zone's sites in the market sit in its pack's frame, or the market's (cities.ts `cityToMarket`)
        const q = worldToCity(cells.xy[2 * c], cells.xy[2 * c + 1], city), p = cityToMarket(q.x, q.y, city);
        xy[2 * i] = p.x;
        xy[2 * i + 1] = p.y;
      });
      return { at, xy, w: Float32Array.from(w) };
    });
    if (!ends[0].at.length && !ends[1].at.length) return;
    const [h, a] = ends;
    const replies = await this.ask((id) => ({ kind: "airLegs", id, version, xy0: h.xy, w0: h.w, xy1: a.xy, w1: a.w }), undefined, [], other, t);
    const sums = ends.map((e) => new Float64Array(3 * e.at.length));
    let mask = 0;
    for (const r of byMask(replies)) {
      if (r.kind !== "airLegs" || !r.legs0 || !r.legs1) return;
      mask |= r.mask;
      [r.legs0, r.legs1].forEach((v, e) => { if (v.length === sums[e].length) addInto(sums[e], v); });
    }
    if (mask !== ALL_PERIODS) return;
    const put: number[] = [], add: number[] = [];
    ends.forEach(({ at }, e) => {
      const s = sums[e], o = 3 * e;
      at.forEach((c, i) => {
        const rail = Math.max(0, s[3 * i + 1]), walk = Math.max(0, s[3 * i + 2]);
        const v = [rail, walk, Math.max(0, s[3 * i] - rail - walk)];
        for (let b = 0; b < 3; b++) { modes[(o + b) * n + c] += v[b]; put.push((o + b) * n + c); add.push(v[b]); }
      });
    });
    if (keep === undefined) return;
    this.lastAir.delete(keep);
    this.lastAir.set(keep, { n: modes.length, at: Int32Array.from(put), add: Float64Array.from(add) });
    for (const k of this.lastAir.keys()) { if (this.lastAir.size <= LAST_AIR_ZONES) break; this.lastAir.delete(k); }
  }
  /** T-4277: what `airportAcross` last added to a zone's bubbles, by zone and day (`city:day`), for while the workers
   * solve the zone again, as `lastReg` is for the rest of its trips with other places */
  private lastAir = new Map<string, { n: number; at: Int32Array; add: Float64Array }>();

  /** T-3503: the trips with other places added to a zone's legs of every purpose (`everydayToo`) or
   * of work alone: under every purpose the airports' from across the zone line too, at their
   * terminals by the Airport purpose's legs. */
  private async withRegional(modes: Float32Array, city: CityId, version: number, other: CityId | undefined, t: Target, everydayToo: boolean): Promise<void> {
    await this.regionalInto(modes, city, version, other, t, everydayToo ? REG_ALL : REG_COMMUTES);
    if (!everydayToo || AIRPORT_KIND < 0) return;
    const key = `${city}:${shownDay.peek()}`;
    // (T-4265: while the workers solve the zone again, the last answer alone: their airport legs would come back empty
    // after a pass)
    if (this.gone(city, version)) {
      const last = this.lastAir.get(key);
      if (last && last.n === modes.length) for (let i = 0; i < last.at.length; i++) modes[last.at[i]] += last.add[i];
      return;
    }
    const [air, cells] = await Promise.all([this.purposeModes("airport", other), this.cityCells(other)]);
    if (!air || !cells) return;
    const w = concat([air.homeRail, air.homeWalk, air.homeDrive, air.workRail, air.workWalk, air.workDrive, air.homeNone, air.workNone]);
    await this.airportAcross(modes, w, cells, city, version, other, t, key);
  }

  /** T-3503: each zone's 600 m squares and its cells' shares of them (static for a pack's cells). */
  private zsq = new Map<CityId, Promise<ZoneSq | null>>();
  private zoneSq(city: CityId, version: number, other: CityId | undefined, auxFirst = false): Promise<ZoneSq | null> {
    let p = this.zsq.get(city);
    if (!p) {
      const ask = async (w: number) => {
        const [r] = await this.ask((id) => ({ kind: "zoneSquares", id, version }), [w], [], other);
        return r?.kind === "zoneSquares" && r.xy && r.sq && r.share ? { xy: r.xy, sq: r.sq, share: r.share } : null;
      };
      // (`auxFirst`, T-4265: the aux worker, never busy, while the solving workers solve the zone again)
      const aux = this.isBare(city) || auxFirst;
      const q: Promise<ZoneSq | null> = (async () => (await ask(aux ? this.auxAt : 0)) ?? (await ask(aux ? 0 : this.auxAt)))();
      this.zsq.set(city, (p = q));
      q.then((v) => { if (!v && this.zsq.get(city) === q) this.zsq.delete(city); });
    }
    return p;
  }
  /** T-3503: the last whole answer per zone, kinds and day, so the bubbles keep their size while a
   * round's is out */
  private lastReg = new Map<string, Float64Array>();
  /**
   * T-3503: the regional layer's trips with other places added to a zone's legs `modes` (eight
   * blocks), at the end of each that is in the zone: longer trips leaving, the residents' commutes
   * and outings at the homes; longer trips arriving, the workers' commutes and the outings coming
   * here at the destinations (airports' instead by `airportAcross`). Each 600 m square takes its 6 km
   * zone's share by people, attraction or jobs (`IntercityApi.zone_legs_day`, as a selection counts
   * them), and its cells their share of it. `kinds` as `regLegs`. Rail is rail, walking walking, the
   * rest (car and plane) driving.
   */
  private async regionalInto(modes: Float32Array, city: CityId, version: number, other: CityId | undefined, t: Target | undefined, kinds: number): Promise<void> {
    const n = modes.length / 8;
    // T-4265: while the workers solve the zone again, the last answer alone (none for a zone not asked about before): they
    // no longer hold the market's figures of `version`, and the question would wait behind a pass for nothing
    const gone = this.gone(city, version);
    if (gone && !this.lastReg.has(`${city}:${kinds}:${shownDay.peek()}`)) return;
    let z = await this.zoneSq(city, version, other, gone);
    if (z && z.sq.length !== n) {
      // the zone was opened again with another set of cells (demandLod.ts)
      this.zsq.delete(city);
      z = await this.zoneSq(city, version, other, gone);
    }
    if (!z || z.sq.length !== n) return;
    const m = z.xy.length / 2, xy = z.xy;
    const replies = gone ? [] : await this.ask((id) => ({ kind: "regLegs", id, version, xy, kinds }), undefined, [], other, t);
    const key = `${city}:${kinds}:${shownDay.peek()}`;
    let sum: Float64Array | undefined = new Float64Array(6 * m), mask = 0;
    for (const r of byMask(replies)) {
      if (r.kind !== "regLegs" || !r.legs) { sum = undefined; break; }
      // an empty answer: no regional market in this worker (a game with no line yet), or none at
      // this moment: the last answer stands, and `lodReset` drops them when a game is loaded
      if (!r.legs.length) { sum = undefined; break; }
      mask |= r.mask;
      if (r.legs.length === sum.length) addInto(sum, r.legs);
    }
    if (sum && mask === ALL_PERIODS) {
      this.lastReg.delete(key);
      this.lastReg.set(key, sum);
      let bytes = 0;
      for (const v of this.lastReg.values()) bytes += v.byteLength;
      for (const k of this.lastReg.keys()) {
        if (bytes <= LAST_REG_BYTES || this.lastReg.size < 2) break;
        bytes -= this.lastReg.get(k)!.byteLength;
        this.lastReg.delete(k);
      }
    } else if ((sum = this.lastReg.get(key))) { this.lastReg.delete(key); this.lastReg.set(key, sum); }
    if (!sum || sum.length !== 6 * m) return;
    for (let c = 0; c < n; c++) {
      const i = z.sq[c];
      for (let end = 0; end < 2; end++) {
        const f = z.share[end * n + c];
        if (!(f > 0)) continue;
        const o = 6 * i + 3 * end, b = 3 * end;
        const legs = sum[o], rail = Math.max(0, sum[o + 1]), walk = Math.max(0, sum[o + 2]);
        if (!(legs > 0)) continue;
        modes[b * n + c] += f * rail;
        modes[(b + 1) * n + c] += f * walk;
        modes[(b + 2) * n + c] += f * Math.max(0, legs - rail - walk);
      }
    }
  }

  /** T-2365: a zone's trips a day by length on the day shown (game/tripLengths.ts), from every
   * solving worker's periods, or for a zone open bare the aux worker's; cached per zone, solve,
   * round and day; null while not known. */
  tripLengths(zone?: CityId, ahead?: Target): Promise<TripLengths | null> {
    this.syncFocus();
    if (!ahead && this.reopening()) return new Promise<void>((r) => this.respawnWaits.push(r)).then(() => this.tripLengths(zone)); // (T-3492, as `viaRound`)
    const dv = demandView.peek();
    if (!dv && !ahead) return Promise.resolve(null);
    const city = zone ?? activeCity.peek(), day = shownDay.peek(), version = dv?.version ?? ahead!.version, round = dv?.round ?? -1;
    // T-2453: as the views' questions are (T-327): while a solve steps through its rounds, of the round
    // after the one shown, answered on reaching it; a zone shown from the focus pool, there
    const at = ahead ?? this.target(city, true) ?? shownTarget(version, round);
    // (a question that waits for a round kept apart from one of a round in hand: a cluster's last
    // round asked ahead comes back empty at the solve's final, and the graph then asks again;
    // T-2476: one that waits is keyed by its solve and round alone, since the view shown moves on
    // meanwhile wherever that solve's own rounds are the view: the focus pool's alone, its round 0 asked
    // as the solve was sent came a round late, 9-20 s)
    const keyAt = (r: number, wait: boolean) => (wait && !perfOff("lengths0") ? `${city}:${day}:${at.version}:${r}:w` : `${city}:${version}:${round}:${day}:${at.version}:${r}:${wait ? "w" : ""}`);
    const key = keyAt(at.round, at.wait);
    const had = this.lengthsP.get(key);
    // T-2453: as `viaRound`: the round shown's answer, when it was asked for in time (`lengthsAhead`),
    // with the next round's asked for at once
    const shown = at.wait && at.shown !== undefined && at.shown >= 0 ? this.lengthsP.get(keyAt(at.shown, true)) : undefined;
    if (had) return shown ?? had;
    const p = this.lengthsRun(city, day, version, at);
    this.lengthsP.set(key, p);
    if (this.lengthsP.size > 12) this.lengthsP.delete(this.lengthsP.keys().next().value!);
    p.then((v) => { if (!v && this.lengthsP.get(key) === p) this.lengthsP.delete(key); });
    return shown ?? p;
  }
  private lengthsP = new Map<string, Promise<TripLengths | null>>();

  /** One trip-lengths question at `at`, its answers added up (`version`: the view it is given for). */
  private lengthsRun(city: CityId, day: DemandDay, version: number, at: Target): Promise<TripLengths | null> {
    return (async (): Promise<TripLengths | null> => {
      // (T-2453: a zone open bare whose figures shown are of no solve, while a solve with it runs: that solve's)
      const bare = this.isBare(city) && !at.solve?.cities.includes(city);
      const replies = bare
        ? await this.ask((id) => ({ kind: "tripLengths", id, version: 0, bare: true }), [this.auxAt], [], city)
        : byMask(await this.ask((id) => ({ kind: "tripLengths", id, version: at.version }), undefined, [], city, at));
      let mask = 0, sum: Float64Array | null = null, edges: Float32Array | null = null, work: Float64Array | null = null, kinds: Float64Array | null = null;
      for (const r of replies) {
        if (r?.kind !== "tripLengths" || !r.commute) return null;
        mask |= r.mask;
        edges = r.edges;
        sum ??= new Float64Array(r.commute.length);
        for (const v of [r.commute, r.everyday]) if (v?.length === sum.length) addInto(sum, v);
        // T-2790: the commutes' trips are the commute answer's first block
        work ??= new Float64Array(edges.length + 1);
        addInto(work, r.commute);
        if (r.kinds) {
          kinds ??= new Float64Array(r.kinds.length);
          if (r.kinds.length === kinds.length) addInto(kinds, r.kinds);
        }
      }
      if (mask !== ALL_PERIODS || !sum || !edges || !work || shownDay.peek() !== day) return null;
      return tripLengthsOf(version, day, edges, sum, work, kinds ?? new Float64Array(0));
    })();
  }

  /** T-2453: the city view's zone has its trip lengths asked for each round as the round starts, so
   * its graph finds the round shown answered when it opens on it (a worker past a round answers
   * nothing of it) */
  private lengthsAhead() {
    const sel = selection.peek();
    if (sel?.kind !== "city" || perfOff("selected")) return;
    const id = sel.city as CityId, f = this.fsolve;
    // T-2476: a focus solve of the zone that has shown no round yet: its round 0, so the graph comes with
    // round 0's figures (asked once they showed, of round 1 by then, it came a round later: 9 s on the
    // Tokyo save)
    if (f && this.fpool?.alive && f.cities[0] === id && f.published < 0 && !perfOff("lengths0"))
      void this.tripLengths(id, { version: f.version, round: 0, view: demandView.peek()?.version ?? f.version, wait: true, solve: f, shown: -1 });
    else void this.tripLengths(id);
  }
  /** T-1533: while a solve's estimates step through the rounds every worker is soon past the round
   * shown, and a question about it comes back empty (a weekend's sums are never pushed): it is asked
   * of the next round instead, which each worker answers on reaching it (T-327). */
  private liveAt(at: Target): Target {
    const t = this.target();
    return t?.wait && t.version === at.version && t.round > at.round ? t : at;
  }

  /** T-4265: the solve whose figures zone `id` shows: its cluster's final (T-392), else the view's */
  private shownOf(id: CityId): number | undefined {
    return this.cityVersion.get(id) ?? demandView.peek()?.version;
  }
  /**
   * T-4265: the bubble sums kept: every round's of the solve `running` (T-2485 drops its old rounds), and each zone's of
   * the final it shows. A whole solve after an edit sets every population of the clusters it solves on the new network,
   * after which no worker holds those zones' figures on screen: their bubbles are drawn from these until its final.
   */
  private keepShownSums(running: number | null) {
    for (const [k, parts] of [...this.pushed]) {
      const [v, r, id] = k.split(":");
      if (Number(v) === running) continue;
      if (!(Number(r) >= ROUNDS && Number(v) === this.shownOf(id as CityId))) this.pushed.delete(k);
      // (one worker's memory however many solving workers sent them)
      else if (parts.length > 1 && everyPeriodOnce(parts)) this.pushed.set(k, [mergeSums(parts)]);
    }
  }
  /** T-4265: no worker holds zone `id`'s figures of solve `version` any more: the newer solve in hand solves its cluster.
   * A question about them would wait behind a worker's pass and come back empty. */
  private gone(id: CityId, version: number): boolean {
    const s = this.cur;
    return !!s && version < s.version && !!s.solving?.has(id);
  }

  /** T-2508: which solve `cellModes` reads a zone's legs from: the focus pool's (and whether settled), a
   * cluster's final (T-392), or the view's. The bubbles are asked again when it changes, also while the
   * view keeps its version (a game load's or an edit's figures put into the result shown before). */
  legsKey(zone: CityId = activeCity.peek()): string {
    const f = this.fpool?.alive ? this.fshown.get(zone) : undefined;
    if (f) return `f${f.s.version}${f.round >= ROUNDS ? "F" : ""}`;
    const src = this.cityVersion.get(zone);
    return src !== undefined ? `c${src}` : `v${demandView.peek()?.version}`;
  }

  cellModes(everydayToo = true, zone?: CityId): Promise<CellModes | null> {
    this.syncFocus();
    // `zone`: a nearby zone's legs (T-276), not cached here
    const other = zone !== undefined && zone !== activeCity.peek() ? zone : undefined;
    const version = demandView.peek()?.version;
    if (version === undefined) return Promise.resolve(null);
    // cached per solve and crowding round (T-084: the dot map asks again for the refined result)
    const day = shownDay.peek();
    // T-2045: a zone shown from the focus pool is asked about there, at the round shown
    const f = this.fpool?.alive ? this.fshown.get(other ?? activeCity.peek()) : undefined;
    const round = f ? f.round : demandView.peek()?.round ?? 0;
    // T-392: a zone shown from a cluster solved ahead of its solve's final is asked about there
    const src = f ? f.s.version : this.cityVersion.get(other ?? activeCity.peek());
    const at = f ? shownTarget(f.s.version, f.round, version) : src !== undefined ? shownTarget(src, ROUNDS, version) : shownTarget(version, round);
    if (everydayToo && other === undefined && this.modesP?.version === version && this.modesP.round === round && this.modesP.day === day && this.modesP.src === src) return this.modesP.p;
    // T-328: the cached result's bubbles, when it kept this zone's for this day
    const kept = this.cachedModes;
    if (version === CACHE_VERSION && everydayToo && other === undefined && day === (kept?.day ?? 0) && kept?.city === activeCity.peek()) {
      return Promise.resolve(cellModesOf(version, day, kept.data, true));
    }
    const p = (async (): Promise<CellModes | null> => {
      const city = other ?? activeCity.peek();
      const tg = other === undefined ? this.liveAt(at) : at;
      if (this.isBare(city)) return this.bareModes(city, everydayToo, version, round, day, other, at.version, tg);
      let mask = 0;
      let sums: Float32Array | null = null;
      let extra: Float32Array | null = null;
      // every worker sent its sums with the round (no question to a worker busy with the next
      // round); asked for only when one is missing. Another day than a weekday's only for the
      // zones on screen when the round was solved (T-1904)
      // (T-2532: an edit's local result's, until its whole solve's are all in)
      // (T-3373 B: under the split, the zone's solving holders alone send theirs)
      // (T-4265: whole when they hold every period once: a final's kept since may be of a pool of another size)
      const full = (k: string) => { const l = (f ? this.fpushed : this.pushed).get(k); return l && everyPeriodOnce(l) ? l : undefined; };
      const sent = full(`${at.version}:${at.round}:${city}:${day}`) ?? full(`${at.version}:${at.round}:${city}:${day}:local`);
      const parts: { mask: number; sums: Float32Array | null; everyday: Float32Array | null }[] = sent ? sent : (await this.ask((id) => ({ kind: "subSums", id, version: at.version }), undefined, [], other, other === undefined ? this.liveAt(at) : at)).map((r) => (r.kind === "subSums" ? r : { mask: 0, sums: null, everyday: null }));
      // in worker (period) order, not arrival order, so the same result gives the same bubbles to
      // the bit (T-328: the cached bubbles are compared with the fresh ones)
      for (const r of [...parts].sort((a, b) => a.mask - b.mask)) {
        if (!r.sums) return null;
        mask |= r.mask;
        if (!sums) sums = r.sums.slice();
        else addInto(sums, r.sums);
        if (r.everyday) {
          if (!extra) extra = r.everyday.slice();
          else addInto(extra, r.everyday);
        }
      }
      if (!sums || mask !== ALL_PERIODS) return null;
      const [r] = await this.ask((id) => ({ kind: "cellModes", id, version: at.version, sums: sums! }), [this.auxAt], [sums.buffer], other, at);
      if (r?.kind !== "cellModes" || !r.modes) return null;
      // (T-1533: a later round of the same solve published meanwhile still takes these: its bubbles ask
      // again, and an older round's beat none while a slow answer chases the rounds)
      if (!stillShown(other, city, version, day)) return null;
      // Bubbles now count trip legs at each home/activity pair, both directions.
      // Old commuter queries count one person; convert before adding everyday travel.
      // (work alone for the purpose filter, T-158: `everydayToo` false)
      for (let k = 0; k < r.modes.length; k++) r.modes[k] = 2 * r.modes[k] + (everydayToo ? extra?.[k] ?? 0 : 0);
      await this.withRegional(r.modes, city, at.version, other, tg, everydayToo);
      if (!stillShown(other, city, version, day)) return null;
      return cellModesOf(version, day, r.modes);
    })();
    if (!everydayToo || other !== undefined) return p; // cached by purposeModes, or a nearby zone's
    this.modesP = { version, round, day, src, p };
    p.then((v) => {
      if (!v && this.modesP?.p === p) this.modesP = null; // not cached when it failed
    });
    return p;
  }

  /** T-325: `cellModes` of a zone open bare, from the aux worker's no-rail pass: the commuters by
   * mode (no rail) and each solving worker's periods' everyday legs, added up as `cellModes` adds
   * the workers' answers for a solved city. */
  private async bareModes(city: CityId, everydayToo: boolean, version: number, round: number, day: DemandDay, other: CityId | undefined, solve: number, t: Target): Promise<CellModes | null> {
    const [r] = await this.ask((id) => ({ kind: "bare", id, version, purpose: everydayToo ? -1 : -2 }), [this.auxAt], [], other);
    if (r?.kind !== "bare" || !r.commute || !r.commute.length) return null;
    let extra: Float32Array | null = null;
    for (const p of bareParts(r)) {
      if (!extra) extra = p.slice();
      else addInto(extra, p);
    }
    if (!stillShown(other, city, version, day) || demandView.peek()?.round !== round) return null;
    const m = r.commute;
    for (let k = 0; k < m.length; k++) m[k] = 2 * m[k] + (everydayToo ? extra?.[k] ?? 0 : 0);
    await this.withRegional(m, city, solve, other, t, everydayToo);
    if (!stillShown(other, city, version, day)) return null;
    return cellModesOf(version, day, m);
  }

  /** A station's riders, or a station complex's (`members`, T-146): every member is asked, and
   * members the demand model merged into one station (they answer identically) count once. */
  stationRiders(stationId: string, top = 20, members: string[] = [stationId]): Promise<StationRiders | null> {
    return this.viaRound(`station|${members.join(",")}|${top}`, (t) => this.stationRidersAt(t, stationId, top, members));
  }
  private async stationRidersAt(t: Target, stationId: string, top: number, members: string[]): Promise<StationRiders | null> {
    const solve = t.solve, version = t.view;
    if (!solve) return null;
    const indices = stationIndices(solve, members);
    if (indices.length === 0) return null;
    const answers = await Promise.all(indices.map((station) => this.ask((id) => ({ kind: "station", id, version: t.version, station }), undefined, [], undefined, t)));
    const kept: Float32Array[] = [], seen = new Set<string>();
    for (const parts of answers) {
      let mask = 0;
      const packed: Float32Array[] = [];
      for (const r of byMask(parts)) {
        if (r.kind !== "station" || !r.packed) break;
        mask |= r.mask;
        packed.push(r.packed);
      }
      if (packed.length !== parts.length) continue; // no answer for this station (none for any: null)
      if (mask !== ALL_PERIODS) return null;
      const sig = packed.map(signature).join("|");
      if (seen.has(sig)) continue;
      seen.add(sig);
      kept.push(...packed);
    }
    const home = new Map<number, number>(), work = new Map<number, number>();
    let toZone: Float32Array | null = null, fromZone: Float32Array | null = null;
    let walked = 0, rode = 0;
    for (const p of kept) {
      const nh = p[0], nw = p[1];
      walked += p[2];
      rode += p[3];
      let o = 4;
      const add = (m: Map<number, number>, n: number) => {
        for (let k = 0; k < n; k++) m.set(p[o + k], (m.get(p[o + k]) ?? 0) + p[o + n + k]);
        o += 2 * n;
      };
      add(home, nh);
      add(work, nw);
      const nz = (p.length - o) / 2;
      const tz = p.subarray(o, o + nz), fz = p.subarray(o + nz, o + 2 * nz);
      if (!toZone || !fromZone) (toZone = tz.slice()), (fromZone = fz.slice());
      else for (let z = 0; z < nz; z++) (toZone[z] += tz[z]), (fromZone[z] += fz[z]);
    }
    if (!toZone || !fromZone || !this.still(t)) return null;
    const cells = (m: Map<number, number>): CellRiders => {
      const t = topOf([...m.keys()], [...m.values()]);
      return { cells: t.keys, riders: t.vals, total: t.vals.reduce((a, b) => a + b, 0) };
    };
    const zones = (v: Float32Array): ZoneRiders => {
      const t = topOf(Uint32Array.from(v, (_, i) => i), v, top);
      return { zones: t.keys, riders: t.vals };
    };
    return { version, stationId, scope: members.join(","), home: cells(home), work: cells(work), walked, rode, homeRidersWorkIn: zones(toZone), workRidersLiveIn: zones(fromZone) };
  }

  /** A station complex's transfers a day (T-176; game/demand.ts `ComplexTransfers`): the route
   * nodes at its members are sent, every worker answers for its periods, and the pairs are added
   * up by station and line at each end (both runs of a line together). */
  transfers(members: string[]): Promise<ComplexTransfers | null> {
    return this.viaRound(`transfers|${members.join(",")}`, (t) => this.transfersAt(t, members));
  }
  private async transfersAt(t: Target, members: string[]): Promise<ComplexTransfers | null> {
    const solve = t.solve, version = t.view, round = t.round;
    if (!solve) return null;
    const inside = new Set(members);
    const nodes: number[] = [];
    const at = routeNodes(solve.layout);
    for (const { o, n, stops, back } of solve.layout) for (let k = 0; k < n; k++) {
      if (inside.has(stops[k])) nodes.push(o + k);
      if (inside.has(back?.[k] ?? stops[k])) nodes.push(o + 2 * n - 1 - k);
    }
    if (nodes.length === 0) return null;
    const parts = await this.transferParts(t, nodes);
    if (!parts) return null;
    const sum = new Map<string, TransferPair>();
    for (const v of parts) {
      for (let k = 0; k + 2 < v.length; k += 3) {
        const a = at[v[k]], b = at[v[k + 1]];
        if (!a || !b) continue;
        const key = `${a.station}|${a.line}|${b.station}|${b.line}`;
        const p = sum.get(key);
        if (p) p.riders += v[k + 2];
        else sum.set(key, { fromStation: a.station, fromLine: a.line, toStation: b.station, toLine: b.line, riders: v[k + 2] });
      }
    }
    if (!this.still(t)) return null;
    return { version, round, scope: members.join(","), pairs: [...sum.values()].sort((x, y) => y.riders - x.riders) };
  }

  /** Every solving worker's transfers with either end at one of `nodes` (`transfers`' answer), in
   * worker (period) order; null unless every period answered. */
  private async transferParts(t: Target, nodes: number[]): Promise<(Float32Array | Float64Array)[] | null> {
    const ids = Uint32Array.from(nodes), parts: (Float32Array | Float64Array)[] = [];
    let mask = 0;
    // T-1816: a zone the network's solve does not cover (panned away) answers nothing, and any zone it covers
    // answers for every line, so a covered one is asked then
    const c = t.solve?.cities, zone = c?.length && !c.includes(activeCity.peek()) ? c[0] : undefined;
    for (const r of byMask(await this.ask((id) => ({ kind: "transfers", id, version: t.version, nodes: ids }), undefined, [], zone, t))) {
      if (r.kind !== "transfers" || !r.pairs) return null;
      mask |= r.mask;
      parts.push(r.pairs);
    }
    return mask === ALL_PERIODS ? parts : null;
  }

  /** Every change between platforms a day (T-554), for the timing of platform crowds:
   * `TrackApi.set_transfers`'s packing, per pair the line, run and stop (in the run's order) a
   * rider leaves, the same where they board, and riders. */
  platformTransfers(): Promise<Float32Array | null> {
    return this.viaRound("platformTransfers", (t) => this.platformTransfersAt(t));
  }
  private async platformTransfersAt(t: Target): Promise<Float32Array | null> {
    const solve = t.solve;
    if (!solve) return null;
    // route node o + i is run 0's stop i, then run 1's stop i - n, each in its run's order
    const at: [number, number, number][] = [];
    for (const { id, o, n } of solve.layout) for (let i = 0; i < 2 * n; i++) at[o + i] = [Number(id), i < n ? 0 : 1, i < n ? i : i - n];
    const nodes: number[] = [];
    at.forEach((_, r) => nodes.push(r));
    if (!nodes.length) return null;
    const parts = await this.transferParts(t, nodes);
    if (!parts) return null;
    const sum = new Map<number, number>();
    for (const v of parts) {
      for (let k = 0; k + 2 < v.length; k += 3) {
        if (!at[v[k]] || !at[v[k + 1]]) continue;
        const key = v[k] * at.length + v[k + 1];
        sum.set(key, (sum.get(key) ?? 0) + v[k + 2]);
      }
    }
    if (!this.still(t)) return null;
    const out = new Float32Array(sum.size * 7);
    let i = 0;
    // (T-3373 B: with exact sums in the pair order, whichever worker added which pair first)
    for (const [key, riders] of SUM_FLAGS.exactSum ? [...sum].sort((x, y) => x[0] - y[0]) : sum) {
      out.set(at[Math.floor(key / at.length)], i);
      out.set(at[key % at.length], i + 3);
      out[i + 6] = riders;
      i += 7;
    }
    return out;
  }

  /** The trip explorer (T-195; game/trip.ts): one trip from `from` to `to` (world metres) as a
   * commute, or as everyday purpose `purpose` (T-586, an index into `EVERYDAY_KEYS`), every period
   * from the worker holding it. Null while there is no solve, or the workers have moved on to a
   * newer network or round. */
  trip(from: [number, number], to: [number, number], purpose = -1): Promise<Trip | null> {
    // T-979: a zone still opening is waited for (a bare one would otherwise be asked of workers
    // that never hold it); a zone open bare has no rounds, and only the aux worker holds it
    const city = activeCity.peek();
    if (!this.isOpen(city)) return this.whenOpen(city).then((ok) => (ok && activeCity.peek() === city ? this.trip(from, to, purpose) : null));
    if (this.isBare(city)) {
      const dv = demandView.peek();
      return dv ? this.tripAt(shownTarget(dv.version, dv.round), from, to, purpose, true) : Promise.resolve(null);
    }
    return this.viaRound(`trip|${from}|${to}|${purpose}`, (t) => this.tripAt(t, from, to, purpose));
  }
  private async tripAt(t: Target, from: [number, number], to: [number, number], purpose: number, bare = false): Promise<Trip | null> {
    const solve = t.solve, version = t.view, round = t.round;
    if (!solve && !bare) return null;
    const at = routeNodes(solve?.layout ?? []);
    const periods: Trip["periods"] = PERIODS.map(() => null);
    let mask = 0, outside = false;
    for (const r of await this.ask((id) => ({ kind: "trip", id, version: t.version, from, to, ...(purpose >= 0 ? { purpose } : {}) }), bare ? [this.auxAt] : undefined, [], undefined, t)) {
      if (r.kind !== "trip" || !r.answers) return null;
      mask |= r.mask;
      r.periods.forEach((q, i) => {
        const v = r.answers![i];
        if (v[0] === -1) outside = true;
        periods[q] = readTripPeriod(v, at);
      });
    }
    if (mask !== ALL_PERIODS || !this.still(t)) return null;
    return { version, round, key: tripKey(from, to), purpose, periods, outside };
  }

  /** T-866: the trip explorer between zones (game/trip.ts `MarketTrip`): the longer trips of a
   * bubble selection's cells with place `far`, or its 6 km zone `farGid` (-1 the whole place). */
  marketTrip(end: "home" | "work", cells: Uint32Array, far: number, farGid: number, near?: { place: number; gid: number }): Promise<(MarketPeriod | null)[] | null> {
    if (cells.length === 0 && !near) return Promise.resolve(null);
    const key = near ? `${near.place}:${near.gid}` : cells.join(",");
    return this.viaRound(`icTrip|${end}|${key}|${far}|${farGid}`, (t) => this.marketTripAt(t, end, cells, far, farGid, near));
  }
  private async marketTripAt(t: Target, end: "home" | "work", cells: Uint32Array, far: number, farGid: number, near?: { place: number; gid: number }): Promise<(MarketPeriod | null)[] | null> {
    const solve = t.solve;
    if (!solve) return null;
    const e = end === "home" ? 0 : 1;
    const sq = near ? undefined : await this.selSquares(t, e, cells);
    if (sq === null) return null;
    const at = routeNodes(solve.layout);
    const periods: (MarketPeriod | null)[] = PERIODS.map(() => null);
    let mask = 0;
    for (const r of await this.ask((id) => ({ kind: "icTrip", id, version: t.version, end: e, cells, far, farGid, ...(sq ? { sq } : {}), ...(near ? { near } : {}) }), undefined, [], undefined, t)) {
      if (r.kind !== "icTrip" || !r.answers) return null;
      mask |= r.mask;
      // T-3632: a worker's unanswered period never replaces another's answer
      r.periods.forEach((q, i) => (periods[q] = readMarketPeriod(r.answers![i], at) ?? periods[q]));
    }
    if (mask !== ALL_PERIODS || !this.still(t)) return null;
    return periods;
  }

  /** Where the commuters of a set of cells go (T-098; game/demand.ts `Flows`): every worker's rail
   * part of its periods, added up, then one worker spreads all modes over the far end. */
  flows(end: "home" | "work", cells: Uint32Array): Promise<Flows | null> {
    if (cells.length === 0) return Promise.resolve(null);
    // a zone open bare has no rounds (T-325)
    if (this.isBare(activeCity.peek())) {
      const dv = demandView.peek();
      return dv ? this.flowsAt(shownTarget(dv.version, dv.round), end, cells) : Promise.resolve(null);
    }
    let sent!: () => void;
    const posted = new Promise<void>((r) => (sent = r));
    const p = this.viaRound(`flows|${end}|${cells.join(",")}`, (t) => this.flowsAt(t, end, cells, sent));
    this.flowsPending = p;
    this.flowsPosted = posted;
    const done = () => {
      sent();
      if (this.flowsPending === p) this.flowsPending = this.flowsPosted = null;
    };
    p.then(done, done);
    return p;
  }
  private async flowsAt(t: Target, end: "home" | "work", cells: Uint32Array, sent?: () => void): Promise<Flows | null> {
    const city = activeCity.peek(), round = demandView.peek()?.round;
    const version = t.view;
    const e = end === "home" ? 0 : 1;
    // the commutes' far end, packed as `flows` answers, and each worker's `purposes` answer
    let p: Float32Array;
    let purposeParts: PurposePart[];
    if (this.isBare(city)) {
      // T-336: a zone open bare: the aux worker's no-rail pass, one part per solving worker's periods
      const [r] = await this.ask((id) => ({ kind: "bareFlows", id, version, end: e, cells }), [this.auxAt]);
      if (r?.kind !== "bareFlows" || !r.packed) return null;
      p = r.packed;
      purposeParts = r.masks.map((mask, i) => ({ mask, work: r.work[i], everyday: r.everyday[i] ?? null, names: r.names }));
    } else {
      const [parts, purposeReplies] = await Promise.all([
        this.ask((id) => ({ kind: "flowRail", id, version: t.version, end: e, cells }), undefined, [], undefined, t),
        this.ask((id) => ({ kind: "purposes", id, version: t.version, end: e, cells }), undefined, [], undefined, t),
      ]);
      let mask = 0;
      let sums: Float32Array | null = null;
      // in worker (period) order, as everywhere the workers' answers are added up (T-336)
      for (const r of byMask(parts)) {
        if (r.kind !== "flowRail" || !r.sums) return null;
        mask |= r.mask;
        if (!sums) sums = r.sums;
        else addInto(sums, r.sums);
      }
      if (!sums || mask !== ALL_PERIODS) return null;
      const asked = this.ask((id) => ({ kind: "flows", id, version: t.version, end: e, cells, sums: sums! }), [0], [sums.buffer]);
      sent?.();
      const [r] = await asked;
      if (r?.kind !== "flows" || !r.packed) return null;
      p = r.packed;
      purposeParts = [];
      // (none: a pool change, or T-3373 B, not every population answered)
      if (!purposeReplies.length) return null;
      for (const reply of purposeReplies) {
        if (reply.kind !== "purposes") return null;
        purposeParts.push(reply);
      }
    }
    // T-1210: added up in the sums worker, and dealt there for the display under each filter
    // (farChunks.ts) with the zone's cells, sent once
    const cc = this.cellsP ? await this.cityCells() : null;
    const key = cc ? `${city}:${worldFrameEpoch.peek()}:${cc.xy.length}` : "";
    const sendXY = !!cc && !this.sumsCells.has(key);
    const a = await this.inSums<FlowsAnswer>({ kind: "flows", packed: p, parts: purposeParts, cells: cc ? { key, ...(sendXY ? { xy: cc.xy } : {}) } : null }, [], () => null);
    if (sendXY && a) {
      this.sumsCells.add(key);
      if (this.sumsCells.size > 4) this.sumsCells.delete(this.sumsCells.values().next().value!);
    }
    const body = a ? a.body : flowsBody(p, purposeParts);
    if (!body) return null;
    const result: Flows = { version, end, ...body };
    for (const [filter, d] of a?.dealt ?? []) primeDealt(result, filter, d);
    if (activeCity.peek() !== city || !this.still(t) || (!t.wait && demandView.peek()?.round !== round)) return null;
    return result;
  }

  /** Every worker's answer to a longer-trip query (each holds some of the periods), in worker
   * (period) order, not arrival order, so the sums are the same every time (T-146 compares them);
   * null unless all five periods answered at the version asked about. */
  private async partsIntercity(kind: "icSel" | "icStation", t: Target, make: (id: number) => Query): Promise<Float64Array[] | null> {
    const replies = await this.ask(make, undefined, [], undefined, t);
    let mask = 0;
    const parts: Float64Array[] = [];
    for (const r of byMask(replies)) {
      if (r.kind !== kind || !r.packed) return null;
      mask |= r.mask;
      parts.push(r.packed);
    }
    if (!this.still(t) || mask !== ALL_PERIODS || !parts.length) return null;
    return parts;
  }

  /** T-1208, T-1210: `msg` worked out in the sums worker (demandSums.worker.ts), the arrays in
   * `transfer` handed over; `inline` does the same here once that worker has failed to start or
   * crashed, and where it fails on a message that handed nothing over. */
  private inSums<T>(msg: SumsAsk, transfer: Transferable[], inline: () => T | null): Promise<T | null> {
    if (!this.sumsWorker && !this.sumsBroken) {
      try {
        const w = new Worker(new URL("./demandSums.worker.ts", import.meta.url), { type: "module" });
        w.onmessage = (e: MessageEvent<FromSums>) => {
          const done = this.sumsWaiting.get(e.data.id);
          this.sumsWaiting.delete(e.data.id);
          if (e.data.error) console.error("demand sums:", e.data.error);
          done?.(e.data.result);
        };
        w.onerror = (e) => {
          console.error("demand sums worker:", e.message);
          this.sumsBroken = true;
          this.sumsWorker = null;
          w.terminate();
          for (const done of this.sumsWaiting.values()) done(null);
          this.sumsWaiting.clear();
        };
        this.sumsWorker = w;
      } catch (err) {
        console.error("demand sums worker:", err);
        this.sumsBroken = true;
      }
    }
    if (!this.sumsWorker) return Promise.resolve(inline());
    const id = this.sumsNext++;
    return new Promise<T | null>((done) => {
      this.sumsWaiting.set(id, done as (v: unknown) => void);
      this.sumsWorker!.postMessage({ ...msg, id } as ToSums, transfer);
    }).then((r) => r ?? (transfer.length ? null : inline()));
  }

  /** The frames and tables a longer-trip answer is read with (`SumsCtx`). */
  private async sumsCtx(own = -1): Promise<SumsCtx> {
    const modes = await ownModes(this.modesBook, own);
    return { market: MARKET_SPEC, world: frameOf(), places: INTERCITY_PLACES.length, own, modes, centres: centresOfPlaces() };
  }

  /** T-2136: the table rows of the market whose book has `key` touching `place` (game/demand.ts
   * `rowsOf`), from the first solving worker to answer with them; null when none holds that market
   * (a cached result shown before the market is open, or a newer one opened). */
  corridorRows(key: string, place: number): Promise<CorridorRows | null> {
    const n = this.computeN;
    if (!n || this.status !== "ready") return Promise.resolve(null);
    const id = this.rowsNext++;
    // (T-3373 B: under the split, the market's holder alone holds it)
    const m = this.holding?.placed(MARKET_UNIT);
    const to = m !== undefined && m < n ? [m] : Array.from({ length: n }, (_, i) => i);
    return new Promise((done) => {
      const t = window.setTimeout(() => this.rowsAnswered(id, null, true), ROWS_WAIT_MS);
      this.rowsAsked.set(id, { left: to.length, done: (r) => { clearTimeout(t); done(r); } });
      for (const i of to) this.workers[i].postMessage({ kind: "rows", id, key, place } satisfies ToDemand);
    });
  }
  private rowsAnswered(id: number, rows: CorridorRows | null, timedOut = false) {
    const w = this.rowsAsked.get(id);
    if (!w) return;
    if (rows || timedOut || --w.left === 0) {
      this.rowsAsked.delete(id);
      w.done(rows);
    }
  }

  /** T-336: a selection of the zone in view, open bare, as the regional market's squares, from the
   * aux worker (the solving workers hold the market but not the zone); null when it has none. */
  private async bareSquares(version: number, end: number, cells: Uint32Array): Promise<SelSquares | null> {
    const [r] = await this.ask((id) => ({ kind: "bareSquares", id, version, end, cells }), [this.auxAt]);
    return r?.kind === "bareSquares" && r.xy && r.share ? { xy: r.xy, share: r.share, ...(r.air ? { air: r.air } : {}) } : null;
  }
  /** A selection of the zone in view as the regional market's squares, for a market question at `t`: a zone open
   * bare's from the aux worker; T-3373 B: under the split, from the zone's solving holder at `t`'s round (the
   * market's holder may not hold the zone); undefined: the worker asked holds the zone, null: none */
  private async selSquares(t: Target, end: number, cells: Uint32Array): Promise<SelSquares | null | undefined> {
    if (this.isBare(activeCity.peek())) return this.bareSquares(t.view, end, cells);
    if (!this.holding || this.fversions.has(t.version)) return undefined;
    const [r] = await this.ask((id) => ({ kind: "bareSquares", id, version: t.version, end, cells }), undefined, [], undefined, t);
    return r?.kind === "bareSquares" && r.xy && r.share ? { xy: r.xy, share: r.share, ...(r.air ? { air: r.air } : {}) } : null;
  }

  /** The longer trips of a bubble selection (game/demand.ts `IntercitySelection`, T-118). */
  /** `still`: whether the selection is still wanted; asked once the far end is on its way, so a
   * superseded one posts nothing (T-2746) */
  intercitySelection(end: "home" | "work", cells: Uint32Array, still?: () => boolean): Promise<IntercitySelection | null> {
    if (cells.length === 0) return Promise.resolve(null);
    return this.viaRound(`icSel|${end}|${cells.join(",")}`, (t) => this.intercitySelectionAt(t, end, cells, still));
  }
  private async intercitySelectionAt(t: Target, end: "home" | "work", cells: Uint32Array, still?: () => boolean): Promise<IntercitySelection | null> {
    const version = t.view;
    const e = end === "home" ? 0 : 1;
    const sq = await this.selSquares(t, e, cells);
    if (sq === null) return null;
    // T-2746: after the far end's `flows` is posted, never before it in worker 0's queue; while a solve
    // runs that can wait for a round, hence the shorter cap then
    const first = this.flowsPosted ?? this.flowsPending;
    if (first) await Promise.race([first, new Promise((r) => setTimeout(r, demandUpdating.peek() ? FLOWS_FIRST_MS : FLOWS_IDLE_MS))]);
    if (still && !still()) return null;
    const parts = await this.partsIntercity("icSel", t, (id) => ({ kind: "icSel", id, version: t.version, end: e, cells, ...(sq ? { sq } : {}) }));
    if (!parts) return null;
    const own = CITIES.find((c) => c.id === activeCity.peek())?.place ?? -1;
    const ctx = await this.sumsCtx(own);
    const r = await this.inSums<Omit<IntercitySelection, "version">>({ kind: "icSel", parts, ctx }, parts.map((p) => p.buffer), () => selectionFlows(sumParts("icSel", parts, ctx.places), ctx));
    return r && { version, ...r };
  }

  /** A station's longer-trip rail riders (game/demand.ts `IntercityStation`, T-118), or a station
   * complex's (`members`, T-146), each demand station once as in `stationRiders`. */
  intercityStation(stationId: string, members: string[] = [stationId]): Promise<IntercityStation | null> {
    return this.viaRound(`icStation|${members.join(",")}`, (t) => this.intercityStationAt(t, stationId, members));
  }
  private async intercityStationAt(t: Target, stationId: string, members: string[]): Promise<IntercityStation | null> {
    const solve = t.solve, version = t.view;
    if (!solve) return null;
    const indices = stationIndices(solve, members);
    if (indices.length === 0) return null;
    const answers = await Promise.all(indices.map((station) => this.partsIntercity("icStation", t, (id) => ({ kind: "icStation", id, version: t.version, station }))));
    const memberParts = answers.filter((a): a is Float64Array[] => !!a);
    if (!memberParts.length) return null;
    const ctx = await this.sumsCtx();
    const r = await this.inSums<Pick<IntercityStation, "starts" | "ends">>({ kind: "icStation", members: memberParts, ctx }, memberParts.flat().map((p) => p.buffer), () => stationFlows(memberParts.map((parts) => sumParts("icStation", parts, ctx.places)), ctx));
    return r && { version, stationId, scope: members.join(","), ...r };
  }

  /** T-1208: `work` once every earlier publish has been applied, so the views change in the order
   * they were published while their sums are worked out in the sums worker. */
  private inOrder(work: () => Promise<void> | void) {
    this.dayQueue = this.dayQueue.then(work).catch((err) => console.error("demand publish:", err));
  }

  /** T-1208: a solve's views of every kind of day for the lines of `layout`, worked out in the sums
   * worker (here where it fails), with the zones open bare among the cities and the corridors' table. */
  private async daysOf(s: Pick<Solve, "version" | "svc">, layout: Solve["layout"], round: number, final: boolean, res: Result[], noCorridors = false): Promise<DayDemand[]> {
    // T-1373: the going-out and coming-back days only while one is near; meanwhile each is the
    // ordinary day's it stands in for (`PLAIN_KIND`). The away day is always made (T-1372).
    const made = madeDays();
    this.madeKey = made.join();
    const input: DayInput = {
      version: s.version, round, final, layout, res: res.map(dayResultOf), dayTable: this.dayTable,
      peakHourFactor: PERIODS.map((_, q) => this.periodTable[q * PERIOD_STRIDE + 5] ?? 1),
      // only the cities this solve covered (T-149: a city not loaded has no view, and its controls wait)
      solved: CITIES.filter(({ id }) => res.every((r) => r.citySummary[id])).map(({ id }) => id),
      places: INTERCITY_PLACES.length, days: made, noCorridors,
    };
    // packed in the worker (`packDays`), worked out whole here where it fails
    const got = (await this.inSums<PackedDays | DayViews[]>({ kind: "days", input }, [], () => dayViews(input)))!;
    const madeViews = Array.isArray(got) ? got : unpackDays(got, layout, res.length);
    const views = DEMAND_DAYS.map((k) => madeViews[made.indexOf(made.includes(k) ? k : PLAIN_KIND[k as DayKind])]);
    const service = new Map(s.svc.lines.map((l) => [l.id, l]));
    // T-2266, T-2286: the riders pay the fares this solve priced them at
    const fares = s.svc.fare.nodes ? s.svc.fare as Fares : undefined;
    const done = new Map<DayViews, DayDemand>();
    return views.map((v, day) => {
      const was = done.get(v);
      if (was) return was;
      // T-1419: the stops each line's figures are for, to carry them onto an edited line; T-1492: and
      // its service, so a panel's running cost moves with the fares these figures give
      for (const l of layout) {
        const d = v.view?.lines.get(l.id), sl = service.get(l.id);
        // (T-3542: per period, as the service's)
        if (d) { d.stops = l.stops; if (sl) { d.tph = Array.from(sl.tph); d.cars = sl.cars; } }
      }
      if (fares) for (const d of v.view?.lines.values() ?? []) d.solvedFares = fares;
      // T-325: and the zones open bare, from their own figures
      for (const id of this.bareIds()) if (!v.cities.has(id)) v.cities.set(id, this.bareView(id, day as DemandDay, v.view));
      const out: DayDemand = { view: v.view, cities: v.cities, intercity: { ...v.intercity, corridors: this.corridorsOf(v.intercity.corridors) } };
      done.set(v, out);
      return out;
    });
  }

  /** The kinds of day the shown views were made for (`daysOf`), joined. */
  private madeKey = "";
  /** T-1373: when a superholiday's rush days come near or go, the shown final result's views again
   * with them or without (the results are kept until a newer solve's final). */
  private refreshDays() {
    const key = madeDays().join();
    if (!this.madeKey || key === this.madeKey) return;
    // (T-4307: a side job's figures were made for the kinds of day before; the main pool's show until its next result)
    this.dropOverlay();
    const shown = this.days[0].view;
    const s = shown?.final ? this.log.find((x) => x.version === shown.version) : undefined;
    const res = s?.results;
    if (!s || !shown || !res?.length || res.some((r) => !r)) return;
    this.madeKey = key;
    const round = shown.round;
    this.inOrder(async () => {
      if (this.days[0].view !== shown) return;
      this.days = await this.daysOf(s, s.layout, round, true, res as Result[]);
    });
  }

  /**
   * T-1008: a kind of day's corridors with the market's book (sent once, shared by every result);
   * none while the book is lost (a newer market's pushed it out), which the workers then send
   * again with their next results.
   */
  private corridorsOf(c: CorridorRail | null): Corridors {
    if (!c) return NO_CORRIDORS;
    const book = this.corridorBooks.get(c.key);
    if (!book) {
      if (!this.tableAsked.has(c.key)) { this.tableAsked.add(c.key); for (const w of this.workers) w.postMessage({ kind: "corridorsAgain" } satisfies ToDemand); }
      return NO_CORRIDORS;
    }
    this.corridorBooks.delete(c.key);
    this.corridorBooks.set(c.key, book);
    const { key: _, ...rail } = c;
    return { book, ...rail };
  }

  /** Every period has reached `round`: build the view the UI reads. */
  private publish(s: Solve, round: number) {
    if (round === 0) s.firstAt = performance.now();
    if (s === this.cur && s.barrier) demandStep.value = { done: round + 1, of: ROUNDS + 1 };
    // T-2485: the bubbles' sums of a round two behind this one are never read again (20 MB a load)
    if (round >= 2) for (const k of [...this.pushed.keys()]) { const [v, r] = k.split(":"); if (Number(v) === s.version && Number(r) < round - 1) this.pushed.delete(k); }
    // T-232: a round before the final one is shown only when nothing final would be held instead
    const final = round >= ROUNDS && !s.rough;
    const res = s.results as Result[];
    if (!final && !s.showEstimates) {
      if (s.newZones) this.fillNewZones(s, round, res);
      // T-324: lines new or changed since the shown result have no figures; theirs come from the
      // first estimate at once, every other figure waits for the final round as above
      if (round === 0 && s === this.cur) this.publishNewLines(s, res);
      if (this.cacheShown()) this.retireFocus(); // (T-2533: a cached result's network solved again)
      return;
    }
    this.inOrder(async () => {
      const days = await this.daysOf(s, s.layout, round, final, res);
      // a cache of the same network shown meanwhile (T-328) holds its estimates back
      if (!final && !s.showEstimates) return;
      if (final) this.sideCovered(s, null); // (T-4220)
      // T-2045: the zones the focus pool has as far on keep its figures until this solve's final
      this.days = this.withFocus(days, s, round, final);
      if (final) finalWorld.value = this.netWorld.get(s.svc) ?? -1;
      this.lodShown(s.cities, this.shownCoarse(s));
      if (final && s === this.cur && !this.refineKept(s)) this.endFocus();
      else this.retireFocus();
      if (final) this.asideIdle(); // (T-4220)
      if (final) {
        s.refinedAt = performance.now();
        // T-3599: the next edit's time left starts from this one's pace and tail
        if (!s.barrier && !s.refresh && s.work && s.lastUnitAt !== undefined) noteEditSolve(s.work, s.refinedAt - s.lastUnitAt, s.localEndAt !== undefined && s.wholeAt !== undefined && s.wholeAt > s.localEndAt ? s.wholeAt - s.localEndAt : null);
        const k = this.uncovered.findIndex((u) => u.svc === s.svc);
        if (k >= 0) {
          this.uncovered.splice(0, k + 1);
          demandBehindFrom.value = this.uncovered[0]?.minute ?? null;
        }
        // an older solve's figures are never read again (its svc and layout are); a few MB each
        for (const o of this.log) if (o.version < s.version) (o.results = []), (o.byRound = []), (o.clusterRes = undefined);
        this.aheadP.clear();
        this.cityVersion.clear();
        this.shownSnap.dropTo(s.version); // (T-4220: what was on screen before this final)
        this.keepShownSums(this.cur !== s ? this.cur?.version ?? null : null);
        if (s === this.cur) {
          // T-4055: a pack that began opening during the solve (a zone at another level, a market) makes it stale:
          // its figures show, and its network waits to be solved again once the pack is open, that solve the final
          const need = this.waiting ? null : this.solveNeed(s.svc);
          const stale = !!need && !this.covers(s, need);
          if (stale) this.waiting = { svc: s.svc, arrivedAt: performance.now() };
          else demandUpdating.value = false;
          demandSolving.value = null;
          this.unsettled = null;
          if (!stale) setTimeout(() => void this.writeCache(s), 0); // T-328; a task of its own (T-789)
          // T-1734: what was held back meanwhile
          if (s.barrier === false || stale) this.flush();
          // T-2560: the levels of detail waited for this solve as for a load's card (`checkHold`)
          if (s.svc === this.cacheSolve) {
            this.cacheSolve = null;
            const now = Date.now();
            for (const id of this.fine) if (this.touched.has(id)) this.touched.set(id, now);
            this.lodSoon();
          }
        }
        const t = this.timeRest;
        if (t?.version === s.version) {
          this.timeRest = undefined;
          if (s.firstAt !== undefined) rememberSolve(Math.round(s.firstAt - s.sentAt), t.stops, t.finalMs, Math.round(s.refinedAt - s.sentAt));
        }
      }
      this.progress();
      this.checkHold();
    });
  }

  /**
   * T-324: the first estimate's figures for the lines the shown result has none for that fit (new
   * lines, and lines whose stops changed), added to the shown result. Everything else shown stays
   * as it is: other lines, the split, the bubbles, and stations a shown line already serves (a
   * station only the new lines serve gets the estimate too). The view keeps the shown solve's
   * version and round, so queries and the bubbles behave as before; the final round replaces all.
   */
  private publishNewLines(s: Solve, res: Result[], cluster?: Uint32Array) {
    this.inOrder(async () => {
      const shown = this.days[0].view;
      const base = this.log.find((x) => x.version === shown?.version);
      if (!base || !shown) return;
      const had = new Map(base.layout.map((l) => [l.id, l.stops.join(",")]));
      // a line given early figures by an earlier solve's estimate is still "new" against the base
      // (T-2793: a rough solve's figures are one zone's riders, only for a line that has no figures yet; a cluster's round
      // has only its own lines' (T-1533), so a line outside would read none; any other line with its shown figures carried
      // onto its new stops keeps them until the final)
      const inRes = cluster ? new Set(Array.from(cluster, (k) => s.layout[k]?.id)) : null;
      const fresh = s.layout.filter((l) => had.get(l.id) !== l.stops.join(",") && (!inRes || inRes.has(l.id)) && (!shown.lines.has(l.id) || (!s.rough && !lineLoadsFor(shown, l.id, "all", l.stops))));
      if (!fresh.length) return;
      s.earlyAt = performance.now();
      s.earlyLines = fresh.map((l) => l.id);
      const ests = await this.daysOf(s, fresh, 0, false, res, true);
      this.days = this.days.map((d, k) => {
        if (!d.view) return d;
        const est = ests[k].view!;
        const lines = new Map(d.view.lines), stations = new Map(d.view.stations);
        // (T-3303: the zone in view's round, longer trips as last solved)
        const takes = new Set(fresh.filter(({ id }) => { const l = est.lines.get(id); return !!l && takesMarketless(d.view!.lines.get(id), l); }).map((l) => l.id));
        for (const id of takes) lines.set(id, { ...est.lines.get(id)!, estimate: true });
        for (const [id, v] of est.stations) if (!stations.has(id)) stations.set(id, v);
        let purposes = d.view.purposes;
        if (est.purposes) {
          purposes = new Map(purposes);
          for (const [key, p] of est.purposes) {
            const old = purposes.get(key);
            const pl = new Map(old?.lines), ps = new Map(old?.stations);
            for (const id of takes) { const l = p.lines.get(id); if (l) pl.set(id, l); }
            for (const [id, v] of p.stations) if (!ps.has(id) && !d.view.stations.has(id)) ps.set(id, v);
            purposes.set(key, { lines: pl, stations: ps });
          }
        }
        return { ...d, view: { ...d.view, lines, stations, purposes } };
      });
    });
  }

  /** T-2793: this solve's round for the zones with no figures shown: their views only. A round's riders per line are
   * partial, and over a final result they would take every line's figures down with them. */
  private fillNewZones(s: Solve, round: number, res: Result[]) {
    this.inOrder(async () => {
      const ests = await this.daysOf(s, [], round, false, res, true);
      if (s !== this.cur) return;
      const filled = (s.filled ??= new Set());
      let any = false;
      const next = this.days.map((d, k) => {
        if (!d.view) return d;
        let cities = d.cities;
        for (const id of s.cities) {
          const v = ests[k]?.cities.get(id);
          if (!v || (d.cities.has(id) && !filled.has(id))) continue;
          if (cities === d.cities) cities = new Map(d.cities);
          cities.set(id, { ...d.view, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay, trainShareByPeriod: v.trainShareByPeriod, split: v.split });
          filled.add(id);
          any = true;
        }
        return cities === d.cities ? d : { ...d, cities };
      });
      if (!any) return;
      this.days = next;
      this.lodShown([...filled], s.coarse);
    });
  }

  /** T-1733: the whole solve after an edit's local results may start (once a version) */
  private wholeGo(s: Solve, early = false) {
    if (s !== this.cur || s.wholeSent || s.endLocal) return;
    s.wholeSent = true;
    for (const w of this.workers.slice(0, this.computeN)) w.postMessage({ kind: "whole", version: s.version, ...(early ? { early: true as const } : {}) } satisfies ToDemand);
  }

  /** T-2188: a city edit's solve ends at its local results (once a version, never after `whole`) */
  private endLocal(s: Solve) {
    if (s !== this.cur || s.wholeSent || s.endLocal) return;
    s.endLocal = true;
    for (const w of this.workers.slice(0, this.computeN)) w.postMessage({ kind: "endLocal", version: s.version } satisfies ToDemand);
  }

  /** T-2188: every solving worker has sent `done` for `s`: a city edit that ended at its local result counts
   * towards the refresh, a solve leaving no local result anywhere (a whole one, the refresh) clears it */
  private solveDone(s: Solve) {
    if (s !== this.cur) return;
    if (!s.localHeld?.slice(0, this.computeN).some(Boolean)) this.localEdits = 0;
    else {
      if (s.endLocal) this.localEdits++;
      // (T-4260: the final result may have been kept before this `done` said it holds local figures)
      if (!this.cachedLocal && !this.waiting) void this.writeCache(s);
    }
    this.refreshSoon();
  }

  /** T-2188: the refresh's timer, from the last edit (`REFRESH_IDLE_MS`); any solve sent clears it */
  private refreshSoon() {
    clearTimeout(this.refreshTimer);
    if (!this.localEdits || perfOff("refresh")) return;
    this.refreshTimer = setTimeout(() => this.refreshNow(), Math.max(0, this.editAt + REFRESH_IDLE_MS - performance.now()));
  }

  /** T-2188: the network shown solved whole again, quietly, while nothing else waits; a game paused with "Recalculate
   * trips while paused" off waits for it to run, as an edit does (T-3427) */
  private refreshNow() {
    const s = this.cur;
    if (!this.localEdits || this.failed) return; // (T-4260: a failed pool does not recover, so there is nothing to wait for)
    // (a network held or the clock held passes: try again, as a pause below does)
    if (!s || s.published < ROUNDS || s.rough || this.waiting || this.hold || gameLoadArmed() || (!display.peek().recalcWhilePaused && !clock.running.peek())) { this.refreshTimer = setTimeout(() => this.refreshNow(), REFRESH_IDLE_MS); return; }
    this.dispatch(s.svc, performance.now(), true);
  }

  /** T-2188: an edit waits for the refresh (T-3925: past 40% of it), which then shows as an edit's whole solve does */
  private showRefresh(s: Solve) {
    if (!s.refresh || s.refresh.shown) return;
    s.refresh.shown = true;
    demandUpdating.value = true;
    demandStep.value = { done: 0, of: 1 };
    demandScope.value = s.cities.length;
    demandSolving.value = s.refresh.solving ? new Set(s.refresh.solving) : null;
    const w = s.work;
    demandWork.value = w && Array.from({ length: this.computeN }, (_, k) => w[k]).every(Boolean) ? w.slice(0, this.computeN) : null;
  }

  /** T-2453: views ask their questions again (`demandAskTurn`) */
  private askTurn() {
    if (perfOff("selected")) return;
    this.lengthsAhead();
    demandAskTurn.value++;
  }

  /** T-2453: `zone` shows no settled figures: a game load's (every figure then is the load's
   * estimates, the focus pool's, or those of a zone left unsettled when the hold ended), or none at all */
  private unsettledZone(zone: CityId): boolean {
    return (!!this.hold || !!this.unsettled?.has(zone) || !cityDemandViews.peek().has(zone)) && !focusZones.peek().get(zone);
  }

  /** T-2453: a round of a cluster solved ahead (T-402) while other figures are shown (the zone in
   * view's from the focus pool): the zone the city view shows, when the cluster has it and it shows
   * nothing settled, takes that round's view of it; the lines and the rest wait for the final.
   * T-2476: so does the zone in view when no focus pool shows it (a small device, a focus pool that
   * stopped), which otherwise showed its figures with no rail until the cluster's final. */
  private async selectedRound(s: Solve, res: Result[], cities: CityId[], round: number) {
    const view = !perfOff("viewRound") && this.viewedSet().has(activeCity.peek()) ? activeCity.peek() : null;
    const want = () => [view, this.selectedZone()].filter((id): id is CityId => !!id && cities.includes(id) && this.unsettledZone(id));
    const ids = want();
    if (!ids.length) return;
    const ests = await this.daysOf(s, [], round, false, res, true);
    if (s !== this.cur) return;
    const now = want().filter((id) => ids.includes(id) && this.cityVersion.get(id) !== s.version);
    if (!now.length) return;
    this.days = this.days.map((d, k) => {
      if (!d.view) return d;
      const cities = new Map(d.cities);
      for (const id of now) {
        const v = ests[k]?.cities.get(id);
        if (v) cities.set(id, { ...d.view, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay, trainShareByPeriod: v.trainShareByPeriod, split: v.split });
      }
      return { ...d, cities };
    });
    this.lodShown(now, s.coarse);
  }

  /**
   * T-392: a cluster's final figures, solved ahead of the rest of the solve, put in the shown result:
   * its lines and their stations, its zones' views, and the regional market's if it is in it. The
   * other figures stay as they are, the network-wide totals among them, until the solve's final;
   * the view keeps the shown version, and questions about the cluster's zones go to this solve
   * (`cityVersion`), whose workers hold it. T-402: `round` below the last comes from a cluster
   * solved ahead in a solve with the barrier.
   */
  /**
   * T-2624: a local result's periods so far, before the rest are in: its lines take the solved periods' figures
   * (`localStep`), and nothing else changes. Only while every solving worker solves locally and `whole` has not
   * gone (else the result may never be whole, and a half shown would stand through the whole solve); never marks
   * a line's figures worked out, a zone's version or the local result shown, which the whole local result does.
   * The same for an edit's whole solve of the zone in view's cluster (`step`) where no local result was shown,
   * each period's final as it lands; the cluster's or the solve's final replaces them.
   */
  private publishStep(s: Solve, e: { published: number; steps?: Step }, res: (Result | null)[], c: { cities: CityId[]; lines: Uint32Array; market: boolean; local?: boolean }) {
    if (s !== this.cur || s.barrier !== false || s.rough || s.refresh || e.published >= ROUNDS) return;
    const local = !!c.local, n = this.computeN;
    const off = () => (local ? s.wholeSent : s.localShownAt !== undefined);
    if (local ? perfOff("steps") : perfOff("wholeSteps")) return;
    if (off() || (local && !Array.from({ length: n }, (_, k) => s.localPlans?.[k]).every((v) => v !== undefined && v > 0))) return;
    const got = res.find((r): r is Result => !!r);
    if (!got) return;
    const solved = res.map(Boolean), full = res.map((r) => r ?? got);
    const st = (e.steps ??= { base: null, asked: [], shown: [] });
    st.asked.push(performance.now());
    const fresh = Array.from(c.lines, (l) => s.layout[l]).filter(Boolean);
    this.inOrder(async () => {
      const shown = this.days[0].view;
      if (s !== this.cur || off() || e.published >= ROUNDS || !shown || shown.version === s.version) return;
      const t0 = performance.now();
      // (the unsolved periods' places hold a solved one's figures, which `localStep` never reads)
      const ests = await this.daysOf(s, fresh, ROUNDS, false, full, true);
      if (s !== this.cur || off() || e.published >= ROUNDS) return;
      const t1 = performance.now();
      const base = (st.base ??= this.days);
      const was = this.log.find((x) => x.version === base[0].view?.version)?.layout ?? [];
      const layoutWas = new Map(was.map((l) => [l.id, l])), layoutNow = new Map(fresh.map((l) => [l.id, l]));
      const levels = PERIODS.map((p) => p.demand), ids = fresh.map((l) => l.id);
      // (the fares rule once: every kind of day's figures of a solve are at its fares; T-2188: a step of a whole solve too,
      // whose unsolved periods keep the shown riders, which would then pay the new fares: a fare edit in a one-city game)
      const held = new Set(ids.filter((id) => { const g = ests[0].view?.lines.get(id); return !!g && !takesMarketless(base[0].view?.lines.get(id), g); }));
      const keep = (id: string) => !held.has(id);
      // (kinds of day standing in for another share its views: merged once)
      const made = new Map<DayDemand, [DayDemand, DayDemand, DayDemand]>();
      const merged = this.days.map((d, k) => {
        const m = made.get(ests[k]);
        if (m && m[0] === d && m[1] === base[k]) return m[2];
        const out = localStep(d, base[k], ests[k], solved, levels, ids, layoutNow, layoutWas, keep);
        made.set(ests[k], [d, base[k], out]);
        return out;
      });
      const t2 = performance.now();
      this.days = this.refineOver(merged, s);
      st.shown.push(performance.now());
      // i18n-ignore
      if (import.meta.env.DEV) console.info(`T2624 v${s.version} step ${st.shown.length} of ${c.cities.slice(0, 4).join(",")}: periods ${solved.flatMap((v, q) => (v ? [q] : [])).join(",")}, views ${Math.round(t1 - t0)} ms, merged ${Math.round(t2 - t1)} ms, shown ${Math.round(performance.now() - t2)} ms`);
    });
  }

  private publishCluster(s: Solve, res: Result[], c: { cities: CityId[]; market: boolean; lines: Uint32Array; local?: boolean }, round: number) {
    if (s !== this.cur) return;
    const final = round >= ROUNDS;
    if (!final && s.newZones) this.fillNewZones(s, round, res);
    // T-1533: after an edit, the zone in view's cluster's free-flow round gives new lines their figures
    if (round === 0 && !final && !s.showEstimates) return this.publishNewLines(s, res, c.lines);
    // T-1790: the zone in view's cluster solved ahead counts its own rounds
    if (s.barrier && c.cities.includes(activeCity.peek())) demandStep.value = { done: round + 1, of: ROUNDS + 1 };
    // T-2453: its zones' questions now go to its next round
    if (s.barrier && !final) this.askTurn();
    const fresh = Array.from(c.lines, (l) => s.layout[l]).filter(Boolean);
    this.inOrder(async () => {
      const shown = this.days[0].view;
      // T-402: with nothing shown yet, or only this solve's own figures of the cluster, a view of the
      // cluster alone, which the solve's own results replace (its totals are the cluster's, so never
      // the solve's final); its rounds before the last only where the solve's estimates are shown
      const alone = !shown || shown.version === s.version;
      if (!final && s.showEstimates && !alone) return this.selectedRound(s, res, c.cities, round);
      if (!final && (!s.showEstimates || !alone)) return;
      if (final) this.sideCovered(s, c.cities); // (T-4220)
      // (a final's zones are its version's before the figures go out: the bubbles' effect asks at once where they
      // come from, T-2508)
      const settle = () => { if (final) for (const id of c.cities) this.cityVersion.set(id, s.version); };
      if (alone) { const days = await this.daysOf(s, fresh, round, false, res); settle(); this.days = this.refineOver(days, s); }
      else {
        const ests = await this.daysOf(s, fresh, round, final, res, !c.market);
        const ids = new Set(s.layout.map((l) => l.id));
        settle();
        const merged = this.days.map((d, k) => {
          if (!d.view) return d;
          const est = ests[k];
          const lines = new Map(d.view.lines), stations = new Map(d.view.stations);
          // lines the network no longer has go too
          for (const id of lines.keys()) if (!ids.has(id)) lines.delete(id);
          // T-3303: a result without the market keeps a shown line solved at other fares
          const held = new Set<string>();
          if (!c.market || c.local) for (const [id, l] of est.view!.lines) if (!takesMarketless(lines.get(id), l)) held.add(id);
          for (const [id, l] of est.view!.lines) if (!held.has(id)) lines.set(id, l);
          for (const [id, v] of est.view!.stations) stations.set(id, v);
          let purposes = d.view.purposes;
          if (est.view!.purposes) {
            purposes = new Map(purposes);
            for (const [key, p] of est.view!.purposes) {
              const old = purposes.get(key);
              const pl = new Map(old?.lines), ps = new Map(old?.stations);
              for (const [id, l] of p.lines) if (!held.has(id)) pl.set(id, l);
              for (const [id, v] of p.stations) ps.set(id, v);
              purposes.set(key, { lines: pl, stations: ps });
            }
          }
          const view = { ...d.view, lines, stations, purposes };
          const cities = new Map<CityId, DemandView>();
          for (const [id, v] of d.cities) cities.set(id, { ...v, lines, stations, purposes });
          for (const id of c.cities) {
            const v = est.cities.get(id);
            if (v) cities.set(id, { ...view, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay, trainShareByPeriod: v.trainShareByPeriod, split: v.split });
          }
          return { view, cities, intercity: c.market ? est.intercity : d.intercity };
        });
        this.days = this.refineOver(merged, s);
      }
      this.lodShown(c.cities, this.shownCoarse(s));
      if (final) {
        for (const id of c.cities) this.cityVersion.set(id, s.version);
        this.keepShownSums(s.version);
        if (s.barrier) this.askTurn(); // T-2453: its zones' questions now go to its final
        // T-2045: a cluster's own final beats the focus pool's figures of its zones (T-2605: not one solved coarse
        // over the zone refined aside)
        const kept = this.asideOver(s) ? this.frefine : null;
        if (c.cities.some((id) => id !== kept && this.fshown.has(id))) {
          const z = new Map(focusZones.peek());
          for (const id of c.cities) if (id !== kept) { this.fshown.delete(id); z.delete(id); }
          focusZones.value = z;
          if (this.frefine && c.cities.includes(this.frefine) && !kept) this.endFocus();
        }
        // T-1733: its lines' riders are worked out (a local result's too, longer trips as last solved)
        linesSettled(fresh.map((l) => l.id), this.netWorld.get(s.svc) ?? -1);
        if (c.local) {
          const first = s.localShownAt === undefined;
          s.localShownAt ??= performance.now();
          if (!s.endLocal) setTimeout(() => this.wholeGo(s), EDIT_SETTLE_MS);
          // T-3616: a network held through a quick local solve goes now
          if (first && this.waiting) setTimeout(() => { if (this.cur === s) this.flush(); }, 0);
        }
        // T-1790: an edit's step 2 (the whole solve) once every local result is in
        if (c.local && s === this.cur && (s.localShown = (s.localShown ?? 0) + 1) >= Math.max(...(s.localPlans ?? [0])) && demandStep.peek().of === 2 && !s.endLocal) {
          demandStep.value = { done: 1, of: 2 };
          demandScope.value = s.cities.length;
        }
        for (const id of c.cities) this.unsettled?.delete(id);
        if (!s.refresh || s.refresh.shown) {
          const left = new Set(demandSolving.peek() ?? []);
          for (const id of c.cities) left.delete(id);
          demandSolving.value = left;
        }
      }
      this.progress();
      this.checkHold();
    });
  }

  /**
   * T-325: a bare zone's view of one kind of day, worked out from its figures exactly as
   * `dayResult` and `dayDemand` work out a solved city's (the figures are what a solve gives it).
   */
  private bareView(id: CityId, day: DemandDay, network: DemandView): DemandView {
    const b = (this.bareOpen.get(id) ?? this.bareKept.get(id))!;
    let trips = 0, rail = 0, walk = 0, none = 0;
    const trainShareByPeriod: [string, number][] = PERIODS.map((period, q) => {
      const s = day !== 0 && this.dayTable.length ? summaryOnDay(b.summary[q], b.parts[q], b.keys, (key) => dayFactor(this.dayTable, day, key, q)) : b.summary[q];
      const [, t, ra, wa] = s;
      trips += t; rail += ra; walk += wa; none += s[13] ?? 0;
      return [period.name, t > 0 ? 100 * ra / t : 0];
    });
    return { ...network, tripsPerDay: trips, railTripsPerDay: rail, trainShareByPeriod, split: splitOf(trips, rail, walk, none) };
  }

  /** Zones with bare figures: open bare, or closed since (T-301). */
  private bareIds(): CityId[] {
    return [...this.bareOpen.keys(), ...this.bareKept.keys()];
  }

  /** T-325: zones opened bare since the shown result join its city views at once (no solve). */
  private addBareViews() {
    const days = this.days;
    let changed = false;
    const next = days.map((d, k) => {
      if (!d.view) return d;
      const add = this.bareIds().filter((id) => !d.cities.has(id));
      if (!add.length) return d;
      changed = true;
      const cities = new Map(d.cities);
      for (const id of add) cities.set(id, this.bareView(id, k as DemandDay, d.view));
      return { ...d, cities };
    });
    if (changed) this.days = next;
  }


}

/** The days the demand views are made for (T-1373): every day, the going-out and coming-back days
 * only while one is near. */
function madeDays(): DemandDay[] {
  return DEMAND_DAYS.filter((k) => (k !== OUT_DAY && k !== BACK_DAY) || rushRun.peek() !== null);
}

let client: DemandClient | null = null;
/** T-3313: the pool's state for a feedback report ("" before it starts) */
export const demandDiagnosis = () => client?.diagnosis() ?? "";
export const demandStuckSince = () => client?.stuckSince() ?? null;
/** T-3329, T-3757: MB of WASM memory each solving worker and the aux worker hold; null before the pool starts */
export function demandWorkersWasmMB(): { solving: number[]; aux: number } | null {
  if (!client) return null;
  const m = client.memory();
  return { solving: m.filter((x) => !x.aux).map((x) => x.wasm), aux: m.filter((x) => x.aux).reduce((a, x) => a + x.wasm, 0) };
}

const TEST = new URLSearchParams(location.search).get("demandTest") === "1";
const DEBUG_LOD = new URLSearchParams(location.search).get("debug") === "1";
/** T-351: the page starts a new game (`?new=1`, or the start menu's New game, T-677): no stations,
 * so the viewed zone opens bare from the start. */
let NEW_GAME = false;
/** T-899: the page opens a saved game (not a new one), whose own view the camera has yet to reach */
let SAVED_GAME = false;

/** Start the pool for a city. With `?demandTest=1` it solves T-005's hand-made network. */
export function startDemand(city = HOME_CITY, newGame = false) {
  if (client) return client;
  NEW_GAME = newGame && !TEST;
  SAVED_GAME = !newGame && !TEST;
  const c = (client = new DemandClient(city));
  onLoadStop((why) => c.fail(why));
  onLoadStuck(() => c.restartStuck()); // T-3781
  setDownloadProbe(() => c.oldestDownload()); // T-2756
  setUnitQuietProbe(() => c.unitQuietMs()); // T-4343
  setTripQuery((from, to, purpose) => c.trip(from, to, purpose)); // T-195
  setMarketTripQuery((end, cells, far, farGid, near) => c.marketTrip(end, cells, far, farGid, near)); // T-866, T-885
  setRowsSource((key, place) => c.corridorRows(key, place)); // T-2136
  setDemandQueries({ transfers: (members) => c.transfers(members), platformTransfers: () => c.platformTransfers(), cityCells: (city) => c.cityCells(city), cellModes: (purpose, city) => c.purposeModes(purpose ?? "all", city), stationRiders: (id, top, members) => c.stationRiders(id, top, members), flows: (end, cells) => c.flows(end, cells), intercitySelection: (end, cells, still) => c.intercitySelection(end, cells, still), intercityStation: (id, members) => c.intercityStation(id, members), tripLengths: (zone) => c.tripLengths(zone), legsKey: (zone) => c.legsKey(zone) });
  if (TEST) c.submitNetwork(demandTestSnapshot());
  if (tracing()) startDemandTrace(c); // T-4206
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twDemand = { addressLimited, poolSize, marketOff, client: c, log: c.log, days: () => demandByDay.peek(), start: c.start, resubmit: () => c.resubmit(), view: () => demandView.peek(), cities: () => cityDemandViews.peek(), nearby: () => nearbyZones.peek(), intercity: () => intercityDemand.peek(), activeCity, cellModes: () => c.cellModes(), stationRiders: (id: string) => c.stationRiders(id), cityCells: () => c.cityCells(), flows: (end: "home" | "work", cells: Uint32Array) => c.flows(end, cells), intercitySelection: (end: "home" | "work", cells: Uint32Array) => c.intercitySelection(end, cells), intercityStation: (id: string) => c.intercityStation(id), transfers: (members: string[]) => c.transfers(members), trip: (from: [number, number], to: [number, number], purpose?: number) => c.trip(from, to, purpose), tripLengths: (zone?: CityId) => c.tripLengths(zone) };
  return c;
}

/** T-889: a loading save's built stations (clockClient's onStationsAhead). */
export function stationsAhead(frame: FrameSpec, xy: Float64Array, fine: string[] | null = null, built: string[][] | null = null) {
  if (!TEST) client?.stationsAhead(frame, xy, fine, built);
}

/** T-1539: a loading save's network key (clockClient's onNetAhead). */
export function netAhead(key: string) {
  if (!TEST) client?.netAhead(key);
}

/** Hand the clock worker's NetworkSnapshot to the pool (from clockClient's onNetworkSnapshot). */
export function submitNetwork(snap: NetworkSnapshot) {
  if (!TEST) client?.submitNetwork(snap);
}

/** T-3869: a save that found no memory has the workers replaced (clockClient's onSaveRoom). */
export function saveRoom() {
  client?.restartForSave();
}

/** T-3628: try the packs that failed to download again now (the tab came back, the browser went online). */
export function retryPackDownloads() {
  client?.retryBroken();
}
