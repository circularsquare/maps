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
//   clock hold once that game's network has its final result (HOLD_ROUND, T-347).
// - The day's five periods are split over the pool by index (worker i solves periods i, i + n,
//   ...). Each worker sends the free-flow estimate of all its periods first, then the crowding
//   rounds.
// - `submitNetwork` takes the clock worker's NetworkSnapshots (already debounced there, SPEC 6.5).
//   With `?demandTest=1` the pool solves T-005's hand-made network instead (workers/demandTest.ts)
//   and ignores the clock's.
// - Never blocks: a newer network supersedes a running solve at the workers' next period
//   boundary; game/demand.ts keeps the last finished result until every period of the new one
//   has landed, then swaps it in whole (free flow first, again after each crowding round).

import { effect } from "@preact/signals";
import { activeCity, CITIES, cityById, getWorldFrame, HOME_CITY, isCity, LEGACY_FRAME, MARKET_FRAME, viewedZone, worldFrameEpoch, type CityId } from "../game/cities";
import type { FrameSpec } from "../game/frame";
import { splitOf } from "../game/types";
import { BACK_DAY, DEMAND_DAYS, OUT_DAY, PLAIN_KIND, rushRun, type DayKind, type DemandDay } from "../game/days";
import { display, selection, shownDay, tool } from "../game/ui";
import {
  cityDemandViews, demandByDay, intercityDemand,
  type DayDemand,
  type DemandView,
  demandStep,
  demandScope,
  demandUpdating,
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
  type CorridorTable,
  NO_CORRIDORS,
  demandSolving,
  roughZones,
  refiningZone,
  focusZones,
  demandAskTurn,
  editsSettling,
} from "../game/demand";
import { FOCUS_WORKERS, FocusPool } from "./demandFocus";
import { deviceGB, smallDevice } from "./deviceMemory";
import { makeFrame } from "../game/frame";
import { choosing, demandIdle, gameLoadArmed, gameLoadSeq, viewFiguresIn, gameWait, loadComplete, loadFailed, loadProgress, markAlive, onLoadStop, startup, takeGameLoadHold, zoneFailures, zoneLoads } from "../game/loading";
import { GAME_MS, GUESS_SHARES, LATER_GUESS, lastWait, loadPace, nextLeft, rememberPace, rememberWait, sameSize, type Pace, sharesOf, solveStage, STAGE_TEXT, stagedFraction, stageInHand, STEP_CAP, timeShare, type StageMs } from "../game/loadBar";
import { linesSettled } from "../game/lineFigures";
import { clock } from "../game/clock";
import { PERIODS, periodsMatch } from "../game/periods";
import { EVERYDAY_KEYS, membersOf, type PurposeFilter } from "../game/purposes";
import { perfOff } from "../perfFlags";
import { LEVEL, marketZonesNear } from "./demandClusters";
import { citiesReached, footprintOf, loadFootprints, REACH_M, zonesNearStations, type Footprint } from "./cityReach";
import { nearbyZones, zoneAt, zonesLoadForView } from "../game/zones";
import { packToLngLat } from "../game/coords";
import { addOpened, reportDemandFine } from "../game/openedZones";
import { builtKeep, firstFine, LEFT_ALONE_MS, LOD_COARSE, LOD_FINE, lodBudgetGB, lodMode, planFine, VIEW_SETTLE_MS, EDIT_KEEP_MS, zonesGB } from "./demandLod";
import { world } from "../game/world";
import { trackZones } from "../game/trackZones";
import { viewCentre } from "../game/listScope";
import { ALL_PERIODS, isQuery, type BareFigures, type DemandNetwork, type FromDemand, type Heap, type OpenRejected, type QueryKind, type ToDemand } from "./demandProtocol";
import { addParts, dayFactor, dayResultOf, dayViews, flowsBody, selectionFlows, signature, stationFlows, summaryOnDay, sumParts, unpackDays, type CorridorRail, type DayInput, type DayViews, type FlowsAnswer, type FromSums, type PackedDays, type PurposePart, type SumsAsk, type SumsCtx, type ToSums } from "./demandSums";
import { primeDealt } from "../game/farChunks";
import { tripLengthsOf, type TripLengths } from "../game/tripLengths";
import { readMarketPeriod, readTripPeriod, setMarketTripQuery, setTripQuery, tripKey, type MarketPeriod, type RouteNodes, type Trip } from "../game/trip";
import { demandTestSnapshot } from "./demandTest";
import type { NetworkSnapshot } from "./protocol";
import { dataUrl } from "../game/dataBase";
import { fetchMarketDrives } from "../game/marketRule";
import { lineFareMul, type Fares } from "../game/fareCurve";
import { saveSlotsState } from "./clockClient";
import { readDemandAhead, readDemandCache, writeDemandCache } from "./saveFile";
import { BUILD, CACHE_FORMAT, CACHE_VERSION, cacheAvailable, canonical, fromStored, sameKey, sha256, storedSize, toStored, type CacheKey, type StoredDemand } from "./demandCache";

/**
 * Crowding rounds after the free-flow estimate (T-022, notes/T-022.md). Each lands as its own
 * update. One round leaves the worst segment at its most overloaded (the first round moves half
 * the riders a penalty pushes, all at once); by the third a player-sized network moves under 1% a
 * round and New York's real network under 10%.
 */
const ROUNDS = 3;
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
 * result of the last time the game was open (T-328) is drawn meanwhile but does not release it.
 * A viewed zone open bare is final from its opening (T-351: `holdDone`).
 */
const HOLD_ROUND = ROUNDS;
const PERIOD_STRIDE = 6;

const FLOWS_FIRST_MS = 1000;

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
 * the next load.
 */
export function poolSize(): number {
  const q = Number(new URLSearchParams(location.search).get("demandWorkers"));
  if (Number.isInteger(q) && q >= 1 && q <= PERIODS.length) return q;
  const hc = navigator.hardwareConcurrency || 4;
  if (display.peek().moreCores) return Math.max(1, Math.min(PERIODS.length, Math.floor(hc / 2) - 1));
  if (smallDevice()) return 1;
  const byThreads = hc >= 12 ? 3 : hc >= 6 ? 2 : 1;
  return deviceGB() === undefined ? Math.min(byThreads, 2) : byThreads;
}

/** The service demand reads, with the UI's ids. */
interface Service {
  frame: FrameSpec;
  fare: { base: number; perKm: number; nodes?: [number, number][]; scales?: Record<string, number>; hsr?: number; ferry?: number; transfer?: number };
  /** `complex`: the station's complex (T-161); absent = a complex of its own */
  stations: { id: string; x: number; y: number; complex?: number; upgrades?: number }[];
  /** `wrapS`: a circle's stand at its first stop, riding on round (T-798) */
  lines: { id: string; stops: string[]; tph: [number, number, number]; hopS: number[][][]; hopKm?: number[][]; fast?: boolean; ferry?: boolean; dwellS: number[][][]; wrapS?: number; cars: number; span?: number[]; catches?: number[][] }[];
  /** station pairs joined by a passage (T-1281) */
  passages?: [string, string][];
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
 * place `own`, 2 per place (`SumsCtx.modes`), from the corridor table's rows: NaN where the table has
 * none (a corridor with no trips, whose bubbles carry none either, T-2122) or the shares are NaN (near
 * places on two landmasses); null with no place or before any are known (all by car). */
const ownModesMade = new WeakMap<CorridorTable, Map<number, Float64Array>>();
function ownModes(t: CorridorTable | null, own: number): Float64Array | null {
  if (!t || own < 0) return null;
  const made = ownModesMade.get(t) ?? ownModesMade.set(t, new Map()).get(t)!;
  let out = made.get(own);
  if (!out) {
    out = new Float64Array(2 * INTERCITY_PLACES.length).fill(NaN);
    const { from, to, air, car } = t;
    for (let j = 0; j < t.n; j++) {
      const p = from[j] === own ? to[j] : to[j] === own ? from[j] : -1;
      if (p >= 0 && p < INTERCITY_PLACES.length) (out[2 * p] = air[j]), (out[2 * p + 1] = car[j]);
    }
    made.set(own, out);
  }
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
/** `CellModes` from the workers' eight blocks a cell (demand.worker.ts `withNone`): rail, walk and
 * drive at the home end, the same at the activity end, then No route at each (T-2215). `copy`
 * slices, else the arrays view `data`. */
function cellModesOf(version: number, day: DemandDay, data: Float32Array, copy = false): CellModes {
  const n = data.length / 8, b = (k: number) => (copy ? data.slice(k * n, (k + 1) * n) : data.subarray(k * n, (k + 1) * n));
  return { version, day, homeRail: b(0), homeWalk: b(1), homeDrive: b(2), workRail: b(3), workWalk: b(4), workDrive: b(5), homeNone: b(6), workNone: b(7) };
}
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
  for (const { id, o, n, stops } of layout) for (let k = 0; k < n; k++) at[o + k] = at[o + 2 * n - 1 - k] = { station: stops[k], line: id, k, o, n };
  return at;
}
/** `members`' indices in the solve's station list, those it has */
const stationIndices = (solve: Solve, members: string[]) => members.map((m) => solve.svc.stations.findIndex((s) => s.id === m)).filter((i) => i >= 0);

/** One solve's bookkeeping, and its timings (window.twDemand.log with ?debug=1). */
interface Solve {
  version: number;
  svc: Service;
  /** per line, the first route node and stop count */
  layout: { id: string; o: number; n: number; stops: string[] }[];
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
  /** T-232: its estimates before the final round are shown (startup, a game load, a city with no
   * figures yet); otherwise the last final result stays shown until this one's final round */
  showEstimates: boolean;
  /** T-1533: a rough solve of the zone in view alone, sent while a whole solve waits for packs; never final */
  rough?: boolean;
  /** T-326: the last worker's clusters kept and solved */
  clusters?: [number, number];
  /** T-124: per solving worker, its periods kept and solved in the clusters it solved */
  periods?: [number, number][];
  /** T-1609: the zones the last worker solved alone against their clusters' frozen loads */
  local?: CityId[];
  /** T-1733: per solving worker, how many clusters it solves locally first (`plan.local`), and
   * whether `whole` went */
  localPlans?: number[];
  wholeSent?: boolean;
  /** T-1790: local cluster results shown so far, and (T-1734) when the first was */
  localShown?: number;
  localShownAt?: number;
  /** solved with the round barrier (estimates by round); else after an edit, counted by stage */
  barrier?: boolean;
  /** T-1734: networks held back while it ran, and when the first came */
  heldFrom?: number;
  /** T-392: each cluster published ahead of the solve's final, by its cities: its results per
   * round and period as they come in, the last round in from every worker, and when its final
   * round was shown */
  clusterRes?: Map<string, { byRound: (Result | null)[][]; published: number; at?: number; roundAt?: number[] }>;
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
/** T-1205 (measured, notes/T-1205.md): a fresh worker's WASM after a solve, as its heap at rest or that
 * solve's heap peak times these, plus MEM_BASE (New York and all-China within 4%) */
const MEM_NEED: [number, number] = [1.38, 1.07];
const MEM_BASE = 64 * 2 ** 20;
/** an idle worker restarts when its WASM is this many times its need and this much above it */
const MEM_OVER = 1.15;
const MEM_FREE = (Number(new URLSearchParams(location.search).get("memFree")) || 384) * 2 ** 20; // ?memFree=MB: checks
/** and the network has been still this long; a pool busy then is looked at again after MEM_RETRY_MS */
const MEM_QUIET_MS = 20_000;
const MEM_RETRY_MS = 5_000;
/** where the browser cannot measure the page: Chrome outside the demand workers, and each demand
 * worker outside its WASM (its JS heap; about 0.7 GB a worker on the all-China save) */
const MEM_OTHER = 0.6 * 2 ** 30;
const MEM_WORKER = 0.7 * 2 ** 30;
const mb = (bytes: number) => Math.round(bytes / 2 ** 20);
/** T-1205: what a game may take in all: about 8 GB (Anita, T-1199), less on a device with less
 * (`deviceGB` reports at most 8) */
function memBudget(): number {
  const dm = deviceGB();
  return (dm === undefined ? 6 : dm >= 8 ? 8 : dm * 0.75) * 2 ** 30;
}
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
  const used = new Set(perfOff("alone") ? s.stations.map((st) => String(st.id)) : s.lines.flatMap((l) => l.stops.map(String)));
  const alone = s.stations.filter((st) => !used.has(String(st.id))).map((st): [number, number] => [st.x, st.y]);
  const passages = (s.passages ?? []).filter(([a, b]) => used.has(String(a)) && used.has(String(b)));
  return {
    frame: s.frame ?? LEGACY_FRAME,
    fare: s.fare ?? { base: 2, perKm: 0 }, // legacy fixed 6 perceived minutes
    stations: s.stations.filter((st) => used.has(String(st.id))).map((st) => ({ id: String(st.id), x: st.x, y: st.y, complex: st.complex, ...(st.upgrades ? { upgrades: st.upgrades } : {}) })),
    lines: s.lines.map((l) => ({ id: String(l.id), stops: l.stops.map(String), tph: l.tph, hopS: l.hopS, hopKm: l.hopKm, ...(l.fast ? { fast: true } : {}), ...(l.ferry ? { ferry: true } : {}), dwellS: l.dwellS, ...(l.wrapS ? { wrapS: l.wrapS } : {}), cars: l.cars, ...(l.span ?{ span: l.span } : {}), ...(l.catches ? { catches: l.catches } : {}) })),
    ...(passages.length ? { passages: passages.map(([a, b]) => [String(a), String(b)] as [string, string]) } : {}),
    ...(alone.length ? { alone } : {}),
  };
}

/** T-1733: after an edit's local results are shown, how long the whole solve waits for the next edit
 * (stations built, then the line, a few seconds apart): until it starts, the next edit is solved locally too */
const EDIT_SETTLE_MS = 8000;

/** Pack a service for the workers; lines naming a station that is not in the list are left out. */
function packService(svc: Service): { net: DemandNetwork; layout: Solve["layout"]; ids: { lines: string[]; stations: string[] } } {
  const index = new Map(svc.stations.map((s, i) => [s.id, i]));
  const lines = svc.lines.filter((l) => l.stops.length >= 2 && l.stops.every((s) => index.has(s)));
  const nStops = lines.reduce((a, l) => a + l.stops.length, 0);
  const net: DemandNetwork = {
    frame: svc.frame,
    stXY: new Float32Array(svc.stations.flatMap((s) => [s.x, s.y])),
    // a station without a complex number is one of its own: numbered past every complex
    complex: new Uint32Array(svc.stations.map((s, i) => s.complex ?? svc.stations.length + i)),
    ...(svc.stations.some((s) => s.upgrades) ? { ups: Uint8Array.from(svc.stations, (s) => s.upgrades ?? 0) } : {}),
    ...(svc.passages?.length ? { passages: Uint32Array.from(svc.passages.filter(([p, q]) => index.has(p) && index.has(q)).flatMap(([p, q]) => [index.get(p)!, index.get(q)!])) } : {}),
    lineN: new Uint32Array(lines.map((l) => l.stops.length)),
    stops: new Uint32Array(nStops),
    times: new Float32Array(6 * nStops),
    tph: new Float32Array(lines.flatMap((l) => l.tph)),
    cars: new Float32Array(lines.map((l) => l.cars)),
    fare: svc.fare,
    hopKm: new Float32Array(2 * nStops),
    span: lines.some((l) => l.span) ? new Float32Array(PERIODS.length * 2 * nStops).fill(1) : new Float32Array(0),
    // T-2196: the ferry fans' lines, only where there are any (`?perfOff=fans`: no fans, for timing)
    ...(lines.some((l) => l.ferry) && !perfOff("fans") ? { ferry: Uint8Array.from(lines, (l) => (l.ferry ? 1 : 0)) } : {}),
  };
  // T-1626, T-2195: a high-speed or ferry line's route nodes charge its multiplier, only where one is not 1
  const muls = lines.map((l) => lineFareMul(l, svc.fare));
  if (muls.some((m) => m !== 1)) {
    const w = new Float32Array(2 * nStops).fill(1);
    let o = 0;
    lines.forEach((l, i) => {
      if (muls[i] !== 1) w.fill(muls[i], 2 * o, 2 * (o + l.stops.length));
      o += l.stops.length;
    });
    net.fareW = w;
  }
  const span = net.span!;
  const layout: Solve["layout"] = [];
  let a = 0;
  for (const l of lines) {
    const n = l.stops.length;
    layout.push({ id: l.id, o: 2 * a, n, stops: l.stops });
    l.stops.forEach((s, k) => (net.stops[a + k] = index.get(s)!));
    for (let lev = 0; lev < 3; lev++) {
      const at = (run: number) => 6 * a + (run * 3 + lev) * n;
      const h0 = l.hopS[0]?.[lev] ?? [], d0 = l.dwellS[0]?.[lev] ?? [];
      const h1 = l.hopS[1]?.[lev] ?? [], d1 = l.dwellS[1]?.[lev] ?? [];
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
  return { net, layout, ids: { lines: layout.map((l) => l.id), stations: svc.stations.map((s) => s.id) } };
}

class DemandClient {
  /** The pool's workers that solve periods, then one more with none (`aux`, demand.worker.ts). */
  private workers: Worker[] = [];
  private computeN = 0;
  private periodsOf: number[][] = [];
  /** T-1607: the periods whose market each solving worker solves in a solve with the barrier */
  private marketsOf: number[][] = [];
  private marketsMsg(i: number) { const m = this.marketsOf[i] ?? []; return { markets: m, share: this.marketsOf.some((qs, j) => qs.join() !== this.periodsOf[j]?.join()) }; }
  /** every worker's sums for a round of the newest solve, sent unasked (T-119): `version:round:city` */
  private pushed = new Map<string, { mask: number; sums: Float32Array; everyday: Float32Array | null }[]>();
  private ready = 0;
  /** demand_periods() from the first worker: PERIOD_STRIDE numbers per period */
  private periodTable: number[] = [];
  private version = 0;
  private cur: Solve | null = null;
  /** T-392: zones whose shown figures come from a cluster published ahead of a newer solve's final
   * (the version they came from); questions about them go to that solve */
  private cityVersion = new Map<CityId, number>();
  /** T-378: the corridor table last sent, whose rows hold the shares of the longer trips not by rail
   * by plane and by car (`ownModes`) */
  private modesTable: CorridorTable | null = null;
  /** T-1208, T-1210: the sums worker (`inSums`), started on first use, and its questions waiting */
  private sumsWorker: Worker | null = null;
  private sumsNext = 1;
  private sumsWaiting = new Map<number, (v: unknown) => void>();
  private sumsBroken = false;
  /** sums worker: the cells it holds the curve ranks of (`flowsAt`), oldest first */
  private sumsCells = new Set<string>();
  /** T-1208: the publishes waiting for their views (`inOrder`) */
  private dayQueue: Promise<void> = Promise.resolve();
  /** T-1008: the market's corridor tables by key, the newest last (`dayCorridors`) */
  private corridorTables = new Map<string, CorridorTable>();
  /** the tables asked for again (`corridorsAgain`) and not back yet */
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

  /** T-1165: a bubble selection's far end being worked out; its longer trips wait for it (up to
   * FLOWS_FIRST_MS), so the market's selection sum (100-200 ms in China) does not hold it up */
  private flowsPending: Promise<unknown> | null = null;
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
   * as last seen, `left` the time left shown and when */
  private job: { items: string[]; cities: CityId[]; startup: boolean; version?: number; best: number; anchor?: { shown: number; frac: number }; since: number; wait: number | null; shares?: StageMs; modelled?: boolean; netAt?: number; dataAt?: number; pace?: Pace; modelData?: number; modelSolve?: number; solveRounds?: { first: number; rest: number; measured: boolean }; firstLand?: number[]; left?: { ms: number; at: number } } | null = null;
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

  // --- T-328: the last final result kept beside the save (workers/demandCache.ts) ---
  private cacheOn = cacheAvailable() && !TEST;
  /** off: no cache is written (a check turns it off to leave a network uncached) */
  cacheWrites = true;
  /** SHA-256 of each pack opened in every worker (and "market"), as it downloads */
  private packHash = new Map<string, { p: Promise<string>; set: (h: string) => void }>();
  /** the zones of the cached result shown, and its network's hash */
  private cachedCities = new Set<CityId>();
  private cachedHash: string | null = null;
  /** its bubbles, if it kept them */
  private cachedModes: StoredDemand["modes"] | null = null;
  /** what the last read and write did (window.twDemand.client.cacheLog with ?debug=1) */
  cacheLog: { at: number; ahead?: boolean; hash?: string; readMs?: number; miss?: string; shownAt?: number; wrote?: { at: number; ms: number; bytes: number; ok: boolean; ahead?: boolean } } | null = null;
  /** every read's `cacheLog`, oldest first */
  cacheLogs: NonNullable<DemandClient["cacheLog"]>[] = [];

  constructor(_city: string) {
    effect(() => { activeCity.value; worldFrameEpoch.value; this.syncFocus(); });
    // T-2045: a zone newly in view during a game load is solved first, in the focus pool
    effect(() => { activeCity.value; viewedZone.value; zonesLoadForView.value; this.focusSoon(); });
    const n = (this.computeN = this.poolN = poolSize());
    solveWaves = Math.ceil(PERIODS.length / n);
    if (new URLSearchParams(location.search).has("debug")) console.info(`demand workers: ${n} (threads ${navigator.hardwareConcurrency}, memory ${(navigator as { deviceMemory?: number }).deviceMemory ?? "not reported"})`);
    for (let i = 0; i <= n; i++) this.periodsOf.push(i < n ? periodsOfWorker(i, n) : []);
    this.marketsOf = marketsOfWorkers(n);
    for (let i = 0; i <= n; i++) this.workers.push(this.spawn(i, i === 0 && n > 1));
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
    // (T-2478: a worker whose script did not download has an error with no message)
    w.onerror = (e) => { const j = at(); if (j >= 0) (/RuntimeError|unreachable|recursive use of an object/.test(e.message ?? "") ? this.workerDied(j, e.message) : this.fail(`worker ${j}: ${e.message ?? "its code could not be downloaded"}`)); };
    w.onmessageerror = () => { const j = at(); if (j >= 0) this.fail(`worker ${j}: a message could not be read`); };
    // T-325: the aux worker is told every solving worker's periods, for the zones it holds bare
    const msg: ToDemand = { kind: "init", packs: [], periods: this.periodsOf[i], ...(i < n ? this.marketsMsg(i) : {}), aux: i === n, warmEveryday, ...(perfOff("inPlace") ? { copies: true } : {}),...(i === n ? { groups: this.periodsOf.slice(0, n) } : {}) };
    w.postMessage(msg);
    return w;
  }

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
    if (this.failed || xy.length < 2) return;
    this.endFocus(); // (a game picked while another loaded)
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
      for (const id of new Set([...reached, ...this.nearOf(svc).markets])) {
        const item = marketItem(id);
        if (this.fetching.has(item) || this.broken.has(item)) continue;
        a.markets.add(id);
        this.fetchMarket(id);
      }
      for (const id of zones) fetchZone(id);
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
    return this.withoutBroken(new Set(viewedZone.peek() && zonesLoadForView.peek() ? [activeCity.peek()] : []));
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

  private lod = lodMode();
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
    if (v && this.viewAt.id === v && now - this.viewAt.at >= VIEW_SETTLE_MS) add(v);
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
    if (this.lod !== "auto") return;
    clearTimeout(this.lodTimer);
    this.lodTimer = setTimeout(() => this.lodCheck(), ms) as unknown as number;
    if (!this.lodTick) this.lodTick = setInterval(() => this.lodCheck(), 30_000) as unknown as number;
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
    if (this.lod !== "auto" || this.failed || this.hold || gameLoadArmed() || !this.lodZones || (this.ahead && !this.ahead.arrived)) return;
    const now = Date.now();
    // T-2180: no level changes while edits settle (the edit note's figures would mix them in), and
    // that time counts as looked at, so a fine zone's ten minutes left alone start after
    if (editsSettling.peek()) {
      for (const id of this.fine) this.touched.set(id, now);
      return;
    }
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
      this.dropCells(id);
      this.fetchCity(id, "full");
    }
    this.lodShown();
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
    return keep;
  }

  /** Close the packs no longer needed (CLOSE_MS, KEEP_SPARE), and bare copies the solves have made
   * redundant. Not before startup has finished nor in a loaded game's hold; `?perfOff=close` never. */
  private sweep() {
    clearTimeout(this.closeTimer);
    this.closeTimer = 0;
    if (perfOff("close") || this.failed || this.ready < this.workers.length || this.held || startup.peek() !== "ready" || this.hold) return;
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
    this.workers[this.computeN].postMessage({ kind: "close", ids, bare: true } satisfies ToDemand);
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
      if (sameBytes(header, again)) return [header, bin];
      if (attempt >= RETRY_MS.length) throw new Error(`${id}.json kept changing while it loaded`);
      await new Promise((r) => setTimeout(r, RETRY_MS[attempt]));
    }
  }

  /** Download one pack file once, counting its bytes as they arrive. */
  private async downloadOnce(file: string, high = false): Promise<Uint8Array> {
    const r = await fetch(file === MARKET_ROWS ? file : dataUrl(file), high ? { priority: "high" } as RequestInit : undefined);
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
    this.workers.forEach((w, i) => {
      const last = i === this.workers.length - 1;
      const pack = open.pack && { ...open.pack, bin: last ? open.pack.bin : open.pack.bin.slice(0) };
      // T-180: a market pack as bytes, each solving worker its own copy (the last solving one the
      // original), decoded there: decoding megabytes and cloning the text six times held the main
      // thread up while the player could already pan. The aux worker never opens one: empty bytes.
      const own = <T extends Uint8Array>(b: T) => (i === this.computeN ? new Uint8Array(0) : i === this.computeN - 1 ? b : b.slice());
      const ms = open.intercity?.map((m) => ({ id: m.id, pack: own(m.bytes) })), r = rows && own(rows);
      const dr = r?.byteLength && this.drives ? this.drives.slice() : undefined;
      const msg: ToDemand = { kind: "open", packs: pack ? [pack] : [], ...(ms ? { intercity: ms } : {}), ...(r?.byteLength ? { rows: r } : {}), ...(dr ? { drives: dr } : {}) };
      const transfer: Transferable[] = pack ? [pack.bin] : [];
      for (const m of ms ?? []) if (m.pack.byteLength) transfer.push(m.pack.buffer as ArrayBuffer);
      if (r?.byteLength) transfer.push(r.buffer as ArrayBuffer);
      if (dr) transfer.push(dr.buffer as ArrayBuffer);
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
    if (mode === "full") this.fetching.add(id);
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
        else this.fetching.add(id);
      }
      this.bytes.set(id, bin.byteLength);
      const own = bin.byteOffset === 0 && bin.byteLength === bin.buffer.byteLength ? bin.buffer as ArrayBuffer : bin.slice().buffer;
      const pack = { id, header: new TextDecoder().decode(header), bin: own };
      // T-328: hashed before the bytes go to the workers (WebCrypto copies them at the call)
      if (!bare) this.hashPack(id, [header, bin]);
      if (!bare) {
        // T-1223: at the level the fine set gives it now
        const level = this.levelFor(id);
        this.levelPosted.set(id, level);
        this.post({ pack: { ...pack, level } });
        return this.postMarkets();
      }
      const msg: ToDemand = { kind: "open", packs: [pack], bare: true, ...(this.firstZones().includes(id) ? { first: true } : {}) };
      this.workers[this.computeN].postMessage(msg, [own]);
    }).catch((err) => { this.unreachable(id, id, `city pack ${id}: ${String(err)}`); this.postMarkets(); });
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
      this.unreachable(item, id, `market pack ${id}: ${String(err)}`);
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
  private marketRows(): Promise<Uint8Array> {
    return (this.rows ??= Promise.all([this.download(MARKET_ROWS), fetchMarketDrives()])
      .then(([rows, drives]) => {
        this.drives = drives;
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
    failures.set(zone, `${why}. ${retry ? "It is tried again within a minute." : "It stays out until the page is reloaded."}`);
    zoneFailures.value = failures;
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
    this.unreachable(item, r.id, `${r.market ? "market pack" : "city pack"} ${r.id} would not open: ${r.error}`, fails < 2);
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
    const n = (this.acks.get(item) ?? 0) + 1;
    this.acks.set(item, n);
    if (n < this.workers.length) return;
    this.opened.add(item);
    if (this.releveling.delete(item as CityId)) this.lodShown();
    if (this.openFails.delete(item)) this.recovered(item);
    this.logLoad(item, false, "openedAt");
    this.settleWaiters();
  }

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
      if (this.waiting && this.cur) this.cur.heldFrom ??= performance.now();
      return;
    }
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
    const coarse = new Set(s.coarse ?? []);
    if ([...need].some((id) => !s.cities.includes(id) || coarse.has(id) !== (this.levelPosted.get(id) === LOD_COARSE)) || [...this.opened].some((item) => isMarket(item) && !s.markets.includes(marketZone(item)))) this.dispatch(s.svc, performance.now());
  }

  /**
   * T-1734: an edit's solve runs to its end: from when it is sent until its final lands, a newer
   * network (an edit, a zone opened or at another level) waits, and is sent the moment the final
   * is shown, solved locally on top of it. The one gap is the moment after the local result shows,
   * until `whole` goes (`EDIT_SETTLE_MS`), so an edit right after another (a line, then its trains)
   * is solved locally too; a network held back since before that waits for the final, or edits faster than
   * the local solve would never let a whole solve start.
   */
  private firm(): boolean {
    const s = this.cur;
    if (!s || s.barrier !== false || s.rough || s.published >= ROUNDS || perfOff("firm")) return false;
    return !(s.localShownAt !== undefined && !s.wholeSent && (s.heldFrom === undefined || s.heldFrom > s.localShownAt));
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
    if (this.fpool?.alive || perfOff("rough") || !need?.has(id) || !this.viewedSet().has(id) || this.ready !== this.workers.length || !this.opened.has(id)) return;
    if (this.rough?.svc === svc && this.rough.id === id) return;
    // a zone with figures is solved rough only for lines new since them, which take its free-flow
    // round's figures as a whole solve's would (T-324); its other figures stay
    const shown = cityDemandViews.peek().has(id);
    if (shown && !this.hasNewLines(svc)) return;
    this.rough = { svc, id };
    const { net, layout, ids } = packService(svc);
    const version = ++this.version;
    this.pushed.clear();
    const now = performance.now();
    this.cur = { version, svc, layout, results: PERIODS.map(() => null), byRound: Array.from({ length: ROUNDS + 1 }, () => PERIODS.map(() => null)), published: -1, arrivedAt: now, sentAt: now, periodMs: [], setupMs: [], memBytes: [], cities: [id], markets: [], coarse: this.coarseOf([id]), showEstimates: !shown, rough: true };
    this.log.push(this.cur);
    if (this.log.length > 50) this.log.shift();
    demandUpdating.value = true;
    demandScope.value = 0;
    const msg: ToDemand = { kind: "solve", version, net, rounds: ROUNDS, barrier: true, first: id, only: [id], ids, ...(perfOff("estimate") ? { exact0: true } : {}) };
    this.postSolve(msg);
  }
  private rough: { svc: Service; id: CityId } | null = null;
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
   * city of it must be covered by the solves (not a zone opened bare). */
  private jobNeedsSolve(job: NonNullable<DemandClient["job"]>): boolean {
    if (job.startup || this.hold) return true;
    const need = this.solveNeed(this.latest());
    return !need || job.cities.some((id) => need.has(id));
  }

  private newJob(atStartup: boolean): NonNullable<DemandClient["job"]> {
    const now = performance.now(), w = gameWait();
    const held = !!this.hold || gameLoadArmed();
    return { items: [], cities: [], startup: atStartup, best: 0, since: held && w.since > 0 ? Math.min(w.since, now) : now, wait: held ? w.seq : null };
  }

  private progressSoon() {
    if (!this.progressTimer) this.progressTimer = setTimeout(() => { this.progressTimer = 0; this.progress(); }, 100) as unknown as number;
  }

  private progress() {
    this.jobProgress();
    this.zoneProgress();
  }

  /** T-343: the best share shown so far for each zone loading outside the load in hand */
  private zoneBest = new Map<CityId, number>();
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
    let moving = false;
    for (const id of [...this.fetching, ...this.fetchingBare, ...this.fetchingUndecided]) {
      if (isMarket(id) || shown.has(id) || next.has(id) || this.unsettled?.has(id) || this.failed) continue;
      let f: number;
      if (this.job?.cities.includes(id)) f = this.job.best;
      else {
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
        const open = this.isOpen(id) ? 1 : log ? Math.min(STEP_CAP, (now - log.downloadedAt!) / openMs) : 0;
        f = (dlMs * dl + openMs * open) / (dlMs + openMs);
        if (!this.isOpen(id)) moving = true;
      }
      const best = Math.max(this.zoneBest.get(id) ?? 0, Math.min(0.99, f));
      next.set(id, best);
    }
    // T-421: a game load's zones still solving after its card went at the view's final, until theirs
    // lands (through any newer solve: they have no final figures yet)
    const s = this.cur;
    if (s && this.unsettled?.size && s.published < ROUNDS && !this.failed) {
      const stops = stopCount(s.svc);
      const f = solveStage(s.sentAt, s.roundAt, s.published, s.byRound[s.published + 1]?.filter(Boolean).length ?? 0, s.roundFirstAt, ROUNDS, solveEstimate(stops, "all"), loadPace().later, now, PERIODS.length, solveWaves).share;
      for (const id of this.unsettled) {
        if (next.has(id)) continue;
        next.set(id, Math.max(this.zoneBest.get(id) ?? 0, Math.min(0.99, f)));
        moving = true;
      }
    }
    this.zoneBest = new Map(next);
    const prev = zoneLoads.peek();
    if (next.size !== prev.size || [...next].some(([id, f]) => prev.get(id) !== f)) zoneLoads.value = next;
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
      if (job.wait === w.seq && s) {
        const net = job.netAt ?? job.since, data = Math.max(net, job.dataAt ?? end);
        rememberWait(w.key, { ms: Math.round(end - job.since), stops: stopCount(s.svc), stages: { game: Math.round(net - job.since), data: Math.round(data - net), demand: Math.round(end - data) } });
        const pace = job.pace ?? loadPace();
        const r = job.solveRounds;
        rememberPace({
          data: job.modelData && data - net > 500 ? (data - net) / job.modelData : pace.data,
          game: net - job.since > 200 ? (net - job.since) / GAME_MS : pace.game,
          solve: job.modelSolve && end - data > 500 ? (end - data) / job.modelSolve : pace.solve,
          later: r?.measured && r.first > 0 ? r.rest / r.first : pace.later,
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
      const ms = isMarket(item) ? OPEN_MARKET_MS / markets + mb * OPEN_MS_PER_MB.market : OPEN_CITY_MS + mb * OPEN_MS_PER_MB.city;
      openMs += ms;
      // T-1195: by the packs in at the solving workers (the aux worker opens far faster); a market pack
      // in counts less, as the market is composed after a batch's last
      openDone += loaded(item) ? ms : ms * Math.min(1, (isMarket(item) ? 0.3 : 0.9) * (this.packAcks.get(item) ?? 0) / this.computeN);
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
    if (needsSolve && s && job.version !== undefined && s.version === job.version) {
      // T-793: a game load's zone in view is final with its cluster solved ahead (T-402): that cluster's rounds
      const a = this.hold && s.ahead && [...this.viewedSet()].every((id) => s.ahead!.cities.includes(id)) ? s.clusterRes?.get(s.ahead.key) : undefined;
      const published = (a ?? s).published, byRound = (a ?? s).byRound;
      const landed = byRound[published + 1]?.filter(Boolean).length ?? 0;
      // (a cluster solved ahead keeps no landing times: seen here, to 0.1 s)
      if (a && landed > 0) (job.firstLand ??= [])[published + 1] ??= now;
      const st = solveStage(s.sentAt, (a ?? s).roundAt, published, landed, a ? job.firstLand : s.roundFirstAt, doneRound, prior, job.pace.later, now, PERIODS.length, solveWaves);
      solveMs = st.ms;
      solve = st.share;
      solveLeft = st.left;
      job.solveRounds = { first: st.first, rest: st.rest, measured: published >= 1 };
    }
    // T-1195: each stage's share of the bar is its expected time: the slot's last load's, else, before
    // the network is in, GUESS_SHARES, then the game stage as it took and the others as modelled. The
    // stage in hand moves by what has landed or its time against its expectation, never past its share.
    // (the model's data time scaled by how this browser's last load compared with it, T-1195)
    const dataMs = dlMs + openMs;
    if (!job.shares || (!job.modelled && netIn)) {
      if (netIn) job.modelData = job.items.every(loaded) ? 0 : dataMs;
      const ms: StageMs | null = was ?? (netIn ? { game: (job.netAt ?? job.since) - job.since, data: job.modelData! * job.pace.data, demand: needsSolve ? prior : 0 } : null);
      job.shares = ms ? sharesOf(ms) : GUESS_SHARES;
      job.modelled = netIn;
    }
    // the data stage by its bytes and the packs open in every worker, or by its time against its expectation
    const openReal = openMs > 0 ? openDone / openMs : 1;
    const dataPart = !netIn && !job.items.length ? 0 : dataMs > 0 ? (dlMs * dl + openMs * openReal) / dataMs : 1;
    const parts: StageMs = {
      game: netIn ? 1 : within(job.since, was?.game ?? GAME_MS * job.pace.game),
      data: job.dataAt !== undefined ? 1 : Math.min(STEP_CAP, Math.max(dataPart, netIn ? within(job.netAt, was?.data ?? dataMs * job.pace.data) : 0)),
      demand: !needsSolve ? 1 : solve,
    };
    const fraction = Math.min(0.99, stagedFraction(job.shares, parts));
    // When the shares change (the network arrives and the guess gives way to the model), the bar keeps
    // its place and the rest of the work is spread over the rest of it.
    let shown = job.anchor ? job.anchor.shown + (1 - job.anchor.shown) * Math.max(0, fraction - job.anchor.frac) / Math.max(1e-6, 1 - job.anchor.frac) : fraction;
    if (shown < job.best - 1e-6) { job.anchor = { shown: job.best, frac: fraction }; shown = job.best; }
    job.best = Math.max(job.best, Math.min(0.99, shown));
    // T-1479: the time left, each stage's expected time less its time so far; a stage running late is
    // guessed to go on a share of its time so far. Unknown on a first load until its network is in.
    // T-1654: always a figure; a late round is estimated from the periods landed (`solveStage`).
    let target: number | null = null;
    if (netIn || was) {
      const sinceGame = now - job.since, gameExp = was?.game ?? GAME_MS * job.pace.game;
      const gameLeft = netIn ? 0 : Math.max(gameExp - sinceGame, 0.2 * sinceGame);
      const dataExp = was?.data ?? dataMs * job.pace.data;
      const sinceData = job.netAt !== undefined ? now - job.netAt : 0;
      const dataLeft = job.dataAt !== undefined ? 0 : !netIn ? dataExp : Math.max(dataExp - sinceData, 0.3 * sinceData * (1 - dataPart));
      target = gameLeft + dataLeft + (needsSolve ? solveLeft : 0);
    }
    if (target !== null) job.left = { ms: nextLeft(job.left?.ms ?? null, target, job.left ? now - job.left.at : 0), at: now };
    const cityMb =job.items.filter((i) => !isMarket(i)).reduce((n, i) => n + (this.bytes.get(i) ?? 0), 0) / 1e6;
    this.barParts = { dlMs, dl, openMs, open, openReal, dataPart, solveMs, solve, prior, stops: stops ?? -1, lines: svc?.lines.length ?? -1, zones: sn?.size ?? -1, items: job.items.length, cityMb, workers: this.computeN, netIn: netIn ? 1 : 0, sinceNet: job.netAt !== undefined ? now - job.netAt : -1, sinceData: job.dataAt !== undefined ? now - job.dataAt : -1, elapsed: now - job.since, published: s && job.version === s.version ? s.published : -2, sent: s && job.version === s.version ? now - s.sentAt : -1,...Object.fromEntries(Object.entries(job.shares).map(([k, v]) => [`share_${k}`, v])), ...Object.fromEntries(Object.entries(parts).map(([k, v]) => [`part_${k}`, v])), was: was ? 1 : 0, fraction, shown: job.best, leftTarget: target ?? -1, left: job.left?.ms ?? -1, modelSolve: job.modelSolve ?? -1, solveLeft }; // ?debug=1
    const cities = job.cities.filter((id) => !isMarket(id));
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
    return { format: CACHE_FORMAT, build: BUILD, workers: this.poolN, flags: new URLSearchParams(location.search).get("perfOff") ?? "", packs: Object.fromEntries(ids.map((id, i) => [id, c.has(id) ? `${hashes[i]}:coarse` : hashes[i]])) };
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
      const cities = Object.keys(stored.packs ?? {}).filter((k) => !isMarket(k));
      if (!need || need.size !== cities.length || cities.some((id) => !need.has(id))) return void (log.miss = "zones");
      const markets = this.wantsMarket(svc);
      const had = Object.keys(stored.packs ?? {}).filter(isMarket).map(marketZone);
      if (had.length !== markets.length || markets.some((id) => !had.includes(id))) return void (log.miss = "zones");
      // (T-1223: the levels the solve will have; before the fine set is made, the save's list's)
      const saved = !this.lodZones && this.lod === "auto" ? this.savedFine : null;
      const coarse = cities.filter((id) => (saved ? !saved.includes(id) : this.levelFor(id) === LOD_COARSE));
      const why = sameKey(stored, await this.keyFor(cities, markets, coarse));
      if (why) return void (log.miss = why);
      // (T-1533: a rough solve's estimate gives way to it)
      if (late() || this.failed) return void (log.miss = "late");
      this.applyCache(svc, canon, stored, hash, cities);
      if (ahead && this.early) this.early.shown = true;
      log.shownAt = performance.now();
    } catch (err) {
      log.miss = "error: " + String(err);
    }
  }

  /** Show a cached result as a final one under CACHE_VERSION, behind the loading card (the clock
   * stays held, T-347); the solve of the same network carries on, its estimates are not shown, and
   * its final round replaces it whole. */
  private applyCache(svc: Service, ids: ReturnType<typeof canonical>, stored: StoredDemand, hash: string, cities: CityId[]) {
    const days = fromStored(stored, ids);
    if (days.length !== DEMAND_DAYS.length) throw new Error("days");
    const now = performance.now();
    // a solve entry for it, so its layout and zones are known as a shown solve's are (T-324, T-325);
    // first in the log, so the newest solve stays last
    this.log.unshift({ version: CACHE_VERSION, svc, layout: packService(svc).layout, results: [], byRound: [], published: stored.round, arrivedAt: now, sentAt: now, periodMs: [], setupMs: [], memBytes: [], cities, markets: Object.keys(stored.packs ?? {}).filter(isMarket).map(marketZone), showEstimates: false });
    this.cachedCities = new Set(cities);
    this.cachedHash = hash;
    this.cachedModes = stored.modes ?? null;
    this.cachedAhead = stored.ahead?.save ?? null;
    if (this.cur?.svc === svc) this.cur.showEstimates = false;
    this.endFocus(); // T-2045: every zone has its final figures
    // T-1533: a rough solve under way gives way, so the packs it holds back open at once (a rough
    // solve of no zone, without the barrier, ends the worker's at its next look)
    if (this.cur?.rough) {
      const version = ++this.version;
      for (const w of this.workers) w.postMessage({ kind: "solve", version, ...(({ net, ids }) => ({ net, ids }))(packService(this.cur.svc)), rounds: ROUNDS, only: [] } satisfies ToDemand);
    }
    demandByDay.value = days;
    this.lodShown(cities, cities.filter((id) => this.levelFor(id) === LOD_COARSE));
    this.addBareViews();
    demandUpdating.value = true; // "Updating demand" until the fresh final
    // T-347: a game load's card and held clock stay until the fresh final lands (T-328 released
    // them here); the card's bar carries on as it was
    this.progress();
  }

  /** The figures shown are the cache's. */
  private cacheShown(): boolean {
    return demandByDay.peek()[0]?.view?.version === CACHE_VERSION;
  }

  /** A final result landed: keep it beside the save (not from a test page or a tab without the slot). */
  private async writeCache(s: Solve) {
    const slots = saveSlotsState.peek();
    if (!this.cacheOn || !this.cacheWrites || !slots.writable || !slots.active) return;
    const days = demandByDay.peek();
    if (days.some((d) => d.view?.version !== s.version)) return;
    const t0 = performance.now();
    const canon = canonical(s.svc);
    const stored = toStored(days, s.cities, canon);
    const [hash, key] = await Promise.all([sha256(canon.text), this.keyFor(s.cities, s.markets, s.coarse ?? [])]);
    // the bubbles, when they were asked for this result (the day shown, every purpose, the viewed zone)
    const m = this.modesP, city = this.focusCity;
    const cm = m && m.version === s.version ? await m.p : null;
    // T-1539: the network the loaded save worked out, so the next load of it shows this at once
    const lk = this.loadKey;
    const ahead = lk?.snapshot && lk.svcKey === keyOf(s.svc) ? { save: lk.key, snapshot: lk.snapshot } : null;
    // the result shown at load is the same one, unless it can now add its bubbles or its network
    if (hash === this.cachedHash && (this.cachedModes || !cm) && (!ahead || this.cachedAhead === ahead.save)) return;
    const modes = cm && { city, data: concat([cm.homeRail, cm.homeWalk, cm.homeDrive, cm.workRail, cm.workWalk, cm.workDrive, cm.homeNone, cm.workNone]), ...(m!.day ? { day: m!.day } : {}) };
    const ms = performance.now() - t0;
    const bytes = storedSize(stored) + (modes?.data.byteLength ?? 0);
    const ok = await this.writeStored(slots.active, hash, { ...key, at: Date.now(), round: s.published, days: stored, ...(modes ? { modes } : {}), ...(ahead ? { ahead } : {}) } satisfies StoredDemand);
    this.cachedHash = hash;
    this.cachedAhead = ahead?.save ?? null;
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
    // T-1008: the days share the market's table, copied once (the clone keeps it shared)
    const tables = new Map<CorridorTable, CorridorTable>();
    const own = <T extends Uint16Array | Uint32Array | Float64Array | Float32Array>(a: T): T => { const b = a.slice() as T; transfer.push(b.buffer as ArrayBuffer); return b; };
    for (const d of v.days) {
      const c = d.intercity?.corridors;
      if (c) {
        let t = tables.get(c.table);
        if (!t) {
          const s = c.table;
          tables.set(s, (t = { key: s.key, n: s.n, from: own(s.from), to: own(s.to), longer: own(s.longer), commuteAt: own(s.commuteAt), commuteLegs: own(s.commuteLegs), air: own(s.air), car: own(s.car) }));
          await new Promise((r) => setTimeout(r, 0));
        }
        const copy: Corridors = { table: t, longerScale: c.longerScale, commuteScale: c.commuteScale, railAt: own(c.railAt), longerRail: own(c.longerRail), commuteRail: own(c.commuteRail) };
        days.push({ ...d, intercity: { ...d.intercity!, corridors: copy } });
      } else days.push(d);
    }
    if (v.modes) transfer.push(v.modes.data.buffer as ArrayBuffer);
    const id = ++this.cacheWriteId;
    return new Promise((resolve) => {
      this.cacheWriteWaits.set(id, resolve);
      this.cacheWriter!.postMessage({ id, slot, hash, value: { ...v, days } }, transfer);
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

  /** A held game's network has its final result (T-347), or its zone in view has (T-421): the clock runs. */
  private checkHold() {
    if (this.hold && this.idle()) return this.settleIdle(); // T-318
    if (this.holdDone()) {
      this.hold = null;
      const s = this.cur!;
      if (s.published < ROUNDS) {
        const solving = demandSolving.peek();
        this.unsettled = new Set(s.cities.filter((id) => solving?.has(id) ?? true));
        // their bars start from their own solve's share (the card's load ended without them)
        for (const id of this.unsettled) this.zoneBest.delete(id);
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
    this.status = "failed: " + why;
    demandUpdating.value = false;

    this.onOpen.splice(0).forEach(w => w.f(false));
    for (const q of this.pendingQueries.values()) q.done([]);
    this.pendingQueries.clear();
    this.endJob();
    this.hold = null;
    this.endFocus();
    zoneLoads.value = new Map();
    console.error("demand:", why);
    loadFailed(why);
  }

  // --- T-1174: a worker whose WASM panicked is replaced ---------------------------------------------

  /** The workers being replaced: `left`, the items a replacement has still to reopen (null before
   * they are sent); `solving`, reopened and sent the solve again; `planned`, restarted to give back
   * memory (T-1205), not after a panic. Each is cleared by its replacement's first `done`. */
  private respawn = new Map<number, { left: Set<string> | null; solving: boolean; planned: boolean }>();
  private respawns = 0;
  /** the next solve has every worker make its market parts afresh (`reopened`, `shrink`) */
  private freshParts = false;

  /** Worker `i`'s WASM panicked (or ran out of memory), leaving its objects unusable: a new worker
   * takes its place, reopens what it held and the solve is sent again. A second panic before a
   * replacement has seen a solve through, or more than MAX_RESPAWNS, stops the pool as before: the
   * cause is then one a fresh worker does not cure (most likely memory, T-1169). `planned`
   * (T-1205): an idle worker restarted to give back its memory, which counts toward neither. */
  private workerDied(i: number, why: string, planned = false) {
    if (this.failed) return;
    const panicked = [...this.respawn.values()].some((r) => !r.planned);
    if (!planned && (panicked || this.respawns >= MAX_RESPAWNS)) return this.fail(`worker ${i}: ${why}`);
    if (planned) console.info(`demand: worker ${i} restarted to give back memory: ${why}`);
    else { console.warn(`demand: worker ${i} panicked, replaced: ${why}`); this.respawns++; }
    this.respawn.set(i, { left: null, solving: false, planned });
    this.workers[i].terminate();
    if (this.isReady.delete(i)) this.ready--;
    this.endQueries();
    this.workers[i] = this.spawn(i, false);
  }

  /** Questions waiting on a worker's answer end empty, as at a failure; the views ask again. */
  private endQueries() {
    for (const q of this.pendingQueries.values()) q.done([]);
    this.pendingQueries.clear();
    this.modesP = null;
    this.purposeP.clear();
    this.aheadP.clear();
  }

  /** The replacement for worker `i` is ready: download again what the worker held (the page keeps no
   * copy; the browser's cache has them) and open it there. */
  private async reopen(i: number) {
    const r = this.respawn.get(i)!, aux = i === this.computeN;
    const items = [...this.fetching].filter((x) => !aux || !isMarket(x));
    const bare = aux ? [...this.fetchingBare] : [];
    const city = async (id: CityId) => {
      const [header, bin] = await this.downloadPack(id);
      return { id, header: new TextDecoder().decode(header), bin: bin.slice().buffer as ArrayBuffer, level: this.levelPosted.get(id) ?? LOD_FINE };
    };
    try {
      const markets = items.filter(isMarket).map(marketZone);
      const [packs, barePacks, intercity, rows, drives] = await Promise.all([
        Promise.all(items.filter((x) => !isMarket(x)).map((id) => city(id as CityId))),
        Promise.all(bare.map(city)),
        Promise.all(markets.map(async (id) => ({ id, pack: await this.download(`${id}.market.json`) }))),
        markets.length ? this.download(MARKET_ROWS) : Promise.resolve(undefined),
        markets.length ? fetchMarketDrives().then((d) => d ?? undefined) : Promise.resolve(undefined),
      ]);
      if (this.respawn.get(i) !== r || this.failed) return;
      // closed meanwhile: left out
      const full = packs.filter((p) => this.fetching.has(p.id)), mk = intercity.filter((m) => this.fetching.has(marketItem(m.id)));
      const bares = barePacks.filter((p) => this.fetchingBare.has(p.id));
      r.left = new Set([...full.map((p) => p.id as string), ...mk.map((m) => marketItem(m.id)), ...bares.map((p) => p.id as string)]);
      const w = this.workers[i];
      if (full.length || mk.length) w.postMessage({ kind: "open", packs: full, ...(mk.length ? { intercity: mk, rows, ...(drives ? { drives } : {}) } : {}) } satisfies ToDemand, [...full.map((p) => p.bin), ...mk.map((m) => m.pack.buffer as ArrayBuffer), ...(mk.length && rows ? [rows.buffer as ArrayBuffer] : []), ...(mk.length && drives ? [drives.buffer as ArrayBuffer] : [])]);
      if (bares.length) w.postMessage({ kind: "open", packs: bares, bare: true } satisfies ToDemand, bares.map((p) => p.bin));
      if (!r.left.size) this.reopened(i);
    } catch (err) {
      if (this.respawn.get(i) === r) this.fail(`worker ${i}: its replacement could not reopen its packs: ${String(err)}`);
    }
  }

  /** The replacement holds what worker `i` held: once every replacement does, the pool is whole again
   * and the newest network is solved again under a new version (the shown figures stay until its final). */
  private reopened(i: number) {
    const r = this.respawn.get(i)!;
    r.left = null;
    r.solving = true;
    if (!this.isReady.has(i)) { this.isReady.add(i); this.ready++; }
    if (this.ready < this.workers.length) return;
    this.status = "ready";
    this.settleWaiters();
    // the replacement composed its market once from every pack, where the others may still hold
    // parts of older compositions (their corridor tables then differ): all make theirs afresh
    this.freshParts = true;
    if (this.waiting) this.flush(true);
    else if (this.cur) this.dispatch(this.cur.svc, performance.now());
    else this.respawn.clear();
  }

  /** `item` is open in no worker. */
  private unack(item: string) {
    this.acks.delete(item);
    for (const by of this.ackedBy) by?.delete(item);
  }

  /** T-1174, dev server only (`?debug=1`, `twDemand.client.debugPanic`): make worker `i` panic. */
  debugPanic(i = 0, when: "now" | "solve" = "now") {
    if (import.meta.env.DEV) this.workers[i]?.postMessage({ kind: "debugPanic", when } satisfies ToDemand);
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
  /** solving workers that have sent `done` for the solve in hand */
  private doneBy = new Set<number>();
  /** a solve after a worker change (`freshParts`) has not published its final yet: questions wait
   * for it, since the new worker holds no earlier result (`target`) */
  private refreshing = false;
  /** the page's memory outside the workers' WASM, from `measureUserAgentSpecificMemory` (null where
   * that is not available: not a cross-origin isolated page) */
  private pageOther: number | null = null;

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

  /** Restart the workers well above their need, together, once the pool is quiet. The replacements
   * reopen what they held and the network is solved again once (`workerDied`'s path); the figures
   * shown stay until that solve's final, the same to the trip. */
  private memCheck() {
    this.memTimer = 0;
    if (this.failed) return;
    const over = this.overNeed();
    if (!over.length) return;
    const wait = this.quietIn();
    if (wait > 0) return this.memSoon(wait);
    for (const o of over) this.workerDied(o.i, `${mb(o.wasm)} MB of WASM, about ${mb(o.need)} MB needed`, true);
  }

  /** Before a solve: fewer solving workers when their projected memory with the rest of the page would
   * pass the budget (`memBudget`). Each holds every open zone and the market, so a worker less saves
   * its whole need, at a slower solve; no figure changes. Never grows back within a page. */
  private capPool() {
    if (this.computeN <= 1 || this.respawn.size || perfOff("cap") || new URLSearchParams(location.search).has("demandWorkers")) return;
    let per = 0;
    for (let i = 0; i < this.computeN; i++) per = Math.max(per, this.needOf(i) ?? 0);
    if (!per) return;
    // (a measured page counts each worker's JS heap in `pageOther`)
    const js = this.pageOther === null ? MEM_WORKER : 0, other = this.pageOther ?? MEM_OTHER;
    const aux = (this.needOf(this.computeN) ?? 0) + js, budget = memBudget();
    let n = this.computeN;
    while (n > 1 && other + n * (per + js) + aux > budget) n--;
    if (n < this.computeN) this.shrink(n, `about ${mb(per + js)} MB a solving worker, ${mb(aux)} MB the aux worker, ${mb(other)} MB the rest of the page, against ${mb(budget)} MB`);
  }

  /** The pool's last solving workers stop and the periods are dealt again among the rest (between
   * solves in each worker); the aux worker moves down to index `n`. */
  private shrink(n: number, why: string) {
    const old = this.computeN, cut = old - n;
    console.info(`demand: ${old} solving workers to ${n}: ${why}`);
    for (let k = n; k < old; k++) this.workers[k].terminate();
    for (const list of [this.workers, this.ackedBy, this.start.workers, this.mem, this.restarted] as unknown[][]) list.splice(n, cut);
    const auxReady = this.isReady.has(old);
    this.isReady = new Set([...this.isReady].filter((i) => i < n));
    if (auxReady) this.isReady.add(n);
    this.ready = this.isReady.size;
    this.acks.clear();
    for (const by of this.ackedBy) for (const item of by ?? []) this.acks.set(item, (this.acks.get(item) ?? 0) + 1);
    this.computeN = n;
    this.periodsOf = [...Array.from({ length: n }, (_, i) => periodsOfWorker(i, n)), []];
    this.marketsOf = marketsOfWorkers(n);
    solveWaves = Math.ceil(PERIODS.length / n);
    this.periodsOf.forEach((periods, i) => this.workers[i].postMessage({ kind: "periods", periods, ...(i < n ? this.marketsMsg(i) : {}), ...(i === n ? { groups: this.periodsOf.slice(0, n) } : {}) } satisfies ToDemand));
    this.endQueries();
    this.freshParts = true;
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
    if (import.meta.env.DEV) for (const i of workers) this.workerDied(i, "a test restart", true);
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
  /** ?debug=1: when each zone opened in the focus pool */
  fopened: Map<CityId, number> | null = null;

  /** A game is loading: start the focus pool (its WASM starts while the save's network is worked out). */
  private startFocus() {
    // (none on a small device: its workers would be a second copy of the zone in the tab's memory, T-2122)
    if (this.fpool || perfOff("focus") || smallDevice() || this.failed || this.cacheShown()) return;
    // (at most FOCUS_WORKERS solving workers: each worker is memory of its own, T-2045's follow-up)
    const n = Math.max(1, Math.min(this.computeN, Number(new URLSearchParams(location.search).get("focusWorkers")) || FOCUS_WORKERS));
    this.fpool = new FocusPool(n, Array.from({ length: n }, (_, i) => periodsOfWorker(i, n)), {
      message: (i, msg) => this.focusMessage(i, msg),
      opened: (id) => { (this.fopened ??= new Map()).set(id, performance.now()); this.focusSoon(); },
      failed: (why) => { console.warn(`demand: the focus pool stopped (${why})`); this.endFocus(); },
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

  /** The pool's workers go (their WASM memory with them); questions still waiting on them get no answer. */
  private stopFocusPool() {
    this.fpool?.end();
    this.fpool = this.fsolve = null;
    this.fpushed.clear();
    this.ffetching.clear();
    for (const [id, q] of this.pendingQueries) if (q.focus) { this.pendingQueries.delete(id); q.done([]); }
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
    if (!p?.alive || perfOff("focusEnd") || !s || s.rough || !s.showEstimates || s.published < 0 || s.published >= ROUNDS || dv?.version !== s.version) return;
    // (a zone the main pool's estimates do not cover, or a solve that has shown nothing yet, keeps it)
    const done = (id: CityId, f: { s: Solve; round: number }) => f.s.svc !== s.svc || (f.round >= 0 && s.cities.includes(id) && (f.round >= ROUNDS || f.round <= s.published));
    const f = this.fsolve;
    if (f && !done(f.cities[0], { s: f, round: f.published })) return;
    for (const [id, v] of this.fshown) if (!done(id, v)) return;
    this.fretiredAt = performance.now();
    this.stopFocusPool();
  }
  /** ?debug=1: when the pool ended ahead of the load's final (`retireFocus`) */
  fretiredAt: number | null = null;

  /** T-2459: the focus pool has `id` of `svc` in hand: being solved, or shown */
  private focusSolves(svc: Service, id: CityId): boolean {
    return !!this.fpool?.alive && ((this.fsolve?.svc === svc && this.fsolve.cities[0] === id) || this.fshown.get(id)?.s.svc === svc);
  }

  /** The zone in view, reached and with no settled figures, is solved in the focus pool. */
  private focusSoon() {
    const p = this.fpool, id = activeCity.peek();
    if (!p?.alive || !viewedZone.peek() || !zonesLoadForView.peek()) return;
    const svc = this.latest();
    if (!svc || !this.solveNeed(svc)?.has(id)) return;
    if (focusZones.peek().get(id) || (this.fsolve?.cities[0] === id && this.fsolve.svc === svc)) return;
    if (!p.holds(id)) {
      if (this.ffetching.has(id)) return;
      this.ffetching.add(id);
      void this.downloadPack(id).then(([header, bin]) => {
        this.ffetching.delete(id);
        if (this.fpool !== p) return;
        const own = bin.byteOffset === 0 && bin.byteLength === bin.buffer.byteLength ? bin.buffer as ArrayBuffer : bin.slice().buffer;
        p.post({ id, header: new TextDecoder().decode(header), bin: own, level: this.levelFor(id) });
      }, () => this.ffetching.delete(id));
      return;
    }
    if (!p.open.includes(id)) return; // its `opened` comes back here
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
    this.lengthsAhead(); // T-2476: the city view's graph of the zone, of round 0
  }

  private focusMessage(i: number, msg: FromDemand) {
    if (this.reply(i, msg)) return;
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
      if (r < ROUNDS) this.fpool?.go(s.version, r + 1);
      this.publishFocus(s, r);
      if (r > 0) s.byRound[r - 1] = [];
    }
  }

  /** A round of the focus solve: the zone's figures and its lines' shown, alone while nothing else is,
   * else put in the shown result, as a cluster solved ahead is (T-402). */
  private publishFocus(s: Solve, round: number) {
    const id = s.cities[0], final = round >= ROUNDS, res = s.results as Result[];
    const lines = s.onlyLines ? Array.from(s.onlyLines, (l) => s.layout[l]).filter(Boolean) : s.layout;
    this.inOrder(async () => {
      const w = this.cur;
      // (the load's own solve further on than a round before the last, or landed)
      if (!this.fpool || (!final && w && !w.rough && w.svc === s.svc && w.published > round) || (w && this.cityVersion.get(id) === w.version)) return void this.retireFocus();
      const ests = await this.daysOf(s, lines, round, false, res, true);
      if (!this.fpool) return;
      const shown = demandByDay.peek()[0].view;
      const alone = !shown || (shown.version === s.version);
      demandByDay.value = alone ? await this.daysOf(s, s.layout, round, false, res) : this.withZone(demandByDay.peek(), ests, [id], false, new Set(s.layout.map((l) => l.id)));
      if (!this.fpool) return;
      this.fshown.set(id, { s, ests, round });
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

  /** The load's own estimates of a round, with the zones shown from the focus pool put back in where
   * the focus solve is as far on (or settled). */
  private withFocus(days: DayDemand[], s: Solve, round: number): DayDemand[] {
    // (T-2495: also once the pool's workers have ended, `retireFocus`)
    if (!this.fshown.size) return days;
    const ids = new Set(s.layout.map((l) => l.id));
    for (const [id, f] of this.fshown) {
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

  /** `days` with the figures of `ests` put in for `zones` (their views) and for the lines and
   * stations `ests` has; the market's if `market`. Lines the network no longer has (`ids`) go. */
  private withZone(days: DayDemand[], ests: DayDemand[], zones: CityId[], market: boolean, ids: Set<string>): DayDemand[] {
    return days.map((d, k) => {
      if (!d.view) return d;
      const est = ests[k];
      const lines = new Map(d.view.lines), stations = new Map(d.view.stations);
      for (const id of lines.keys()) if (!ids.has(id)) lines.delete(id);
      for (const [id, l] of est.view!.lines) lines.set(id, l);
      for (const [id, v] of est.view!.stations) stations.set(id, v);
      let purposes = d.view.purposes;
      if (est.view!.purposes) {
        purposes = new Map(purposes);
        for (const [key, p] of est.view!.purposes) {
          const old = purposes.get(key);
          const pl = new Map(old?.lines), ps = new Map(old?.stations);
          for (const [id, l] of p.lines) pl.set(id, l);
          for (const [id, v] of p.stations) ps.set(id, v);
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

  private receive(i: number, msg: FromDemand) {
    if (msg.kind === "failed") return this.fail(`worker ${i}: ${msg.error}`);
    if (msg.kind === "dead") return this.workerDied(i, `${msg.error}${msg.panic ? ` (${msg.panic})` : ""}, ${Math.round(msg.memBytes / 2 ** 20)} MB of WASM memory`);
    if (msg.kind === "packOpened") {
      const item = msg.market ? marketItem(msg.id) : msg.id;
      if (i < this.computeN) this.packAcks.set(item, (this.packAcks.get(item) ?? 0) + 1);
      return this.progressSoon();
    }
    if (this.reply(i, msg)) return;
    markAlive(); // T-950: (not a query's answer, which an idle worker gives while another is stuck)
    if (msg.kind === "loads") {
      // T-1607: on to the worker solving the period's other side
      const to = msg.side === "cities" ? this.marketsOf.findIndex((qs) => qs.includes(msg.period)) : this.periodsOf.findIndex((qs, j) => j < this.computeN && qs.includes(msg.period));
      if (to >= 0 && to !== i) this.workers[to].postMessage(msg, [msg.loads.buffer]);
      return;
    }
    if (msg.kind === "sums") {
      const key = `${msg.version}:${msg.round}:${msg.city}:${msg.day ?? 0}`;
      const list = this.pushed.get(key);
      if (list) list.push({ mask: msg.mask, sums: msg.sums, everyday: msg.everyday });
      else if (this.cur && msg.version === this.cur.version) this.pushed.set(key, [{ mask: msg.mask, sums: msg.sums, everyday: msg.everyday }]);
      return;
    }
    if (msg.kind === "ready") {
      this.start.workers[i] = { wasmMs: msg.wasmMs, openMs: msg.openMs, warmMs: msg.warmMs, memBytes: msg.memBytes };
      this.noteMem(i, msg.memBytes, msg.heap);
      // T-173: worker 0 has warmed the everyday code up; the held city packs may go
      if (i === 0 && this.held) { const held = this.held; this.held = null; held.forEach((send) => send()); this.postMarkets(); }
      if (!periodsMatch(msg.periods, PERIOD_STRIDE))
        console.error("demand: game/periods.ts disagrees with sim/src/track/params.rs PERIOD_HOURS / PERIOD_LEVEL", msg.periods);
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
      for (const id of Object.keys(msg.cityInfo)) this.openedBy(id, i);
      for (const id of msg.markets) this.openedBy(marketItem(id), i);
      if (left && !left.size) this.reopened(i);
      this.flush();
      this.progress();
      return;
    }
    // T-1008: a worker sends each corridor table once, maybe on a result superseded here
    if (msg.kind === "result" && msg.corridorTable) {
      const t = msg.corridorTable;
      this.corridorTables.delete(t.key);
      this.corridorTables.set(t.key, t);
      this.tableAsked.delete(t.key);
      // more than one is in use only while the market changes (`dayCorridors` keeps its own newest)
      if (this.corridorTables.size > 3) this.corridorTables.delete(this.corridorTables.keys().next().value!);
      this.modesTable = t;
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
      return;
    }
    if (msg.kind === "plan") {
      if (i === 0) {
        demandSolving.value = new Set(msg.solving.flat());
        if (msg.ahead) s.ahead = { cities: msg.ahead, key: msg.ahead.join(",") };
        // (T-2459: the focus pool's final of the zone in view may already hold the clock's release)
        if (msg.ahead && s.focused) { this.progress(); this.checkHold(); }
      }
      // T-1733: a worker that solves nothing locally first starts its whole solve at once, so the
      // local result can never be shown whole: the others go on too
      const plans = (s.localPlans ??= []);
      plans[i] = msg.local ?? 0;
      const n = this.computeN, all = plans.slice(0, n);
      if (all.filter((v) => v !== undefined).length === n && all.some((v) => v > 0)) {
        if (all.some((v) => !v)) this.wholeGo(s);
        else if (s === this.cur && !s.barrier) {
          demandStep.value = { done: 0, of: 2 };
          demandScope.value = 0;
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
      if (!res.every(Boolean) || round <= e.published) return;
      e.published = round;
      (e.roundAt ??= [])[round] = performance.now();
      if (round > 0) e.byRound[round - 1] = [];
      if (round < ROUNDS && s.ahead?.key === key) for (const w of this.workers.slice(0, this.computeN)) w.postMessage({ kind: "go", version: s.version, round: round + 1, group: 0 } satisfies ToDemand);
      if (round >= ROUNDS) e.at = performance.now();
      this.publishCluster(s, res as Result[], msg.cluster, round);
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
      if (round === ROUNDS && !s.rough) { this.refreshing = false; this.lastStir = performance.now(); this.memSoon(); void this.measurePage(); }
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
    let svc = serviceFromSnapshot(snap);
    const key = keyOf(svc);
    // the first network after a whole-game load is the one the clock waits for (T-149)
    const gameLoad = takeGameLoadHold();
    if (gameLoad) this.hold = {};
    if (gameLoad && this.ahead) this.ahead.arrived = true;
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
    if (key === this.lastKey) {
      // nothing demand reads changed: the solve in hand (or waiting) is the loaded game's; stations
      // built or removed on no line may reach other zones (T-2410)
      const held = this.latest();
      if (held && held !== svc && this.setAlone(held, svc.alone)) this.ensure();
      if (this.hold && !this.waiting) this.hold.version = this.cur?.version;
      this.checkHold();
      return;
    }
    this.lastKey = key;
    this.lastStir = performance.now();
    if (gameLoad) this.uncovered = [];
    else this.uncovered.push({ svc, minute: clock.minute.peek() });
    if (this.uncovered.length > 50) this.uncovered.splice(1, 1); // (the oldest's minute is what counts)
    demandBehindFrom.value = this.uncovered[0]?.minute ?? null;
    // T-2045: the focus pool is a game load's; an edit's network ends it
    if (!gameLoad) this.endFocus();
    else if (svc.stations.length) this.startFocus();
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
      this.job = { ...this.newJob(startup.peek() === "loading"), ...(same ? { best: this.job.best, anchor: this.job.anchor, since: this.job.since, shares: this.job.shares } : {}) };
    }
    this.send(svc, performance.now());
    this.focusSoon();
    // T-328: a whole game's network: its last final result may be kept beside the save (shown
    // already if it was found ahead of the network, T-1539)
    if (this.hold && this.cacheOn && !(early?.shown && svc === early.svc)) void this.tryCache(svc);
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
    if (!this.cacheShown()) return;
    this.cachedCities = new Set();
    this.cachedHash = this.cachedAhead = null;
    this.cachedModes = null;
    demandByDay.value = DEMAND_DAYS.map(() => ({ view: null, cities: new Map(), intercity: null }));
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

  private dispatch(svc: Service, arrivedAt: number) {
    this.capPool();
    this.doneBy.clear();
    if (this.freshParts) this.refreshing = true;
    const { net, layout, ids } = packService(svc);
    const version = ++this.version;
    if (this.hold) this.hold.version = version;
    const cities = CITIES.map((c) => c.id).filter((id) => this.opened.has(id));
    // the load in hand ends with this solve if it covers every city of it the solves must (T-325:
    // a zone opened bare needs none)
    const need = this.solveNeed(svc);
    if (this.job && this.jobNeedsSolve(this.job) && this.job.cities.every((id) => cities.includes(id) || (need !== null && !need.has(id)))) this.job.version = version;
    this.pushed.clear();
    // T-232: a re-solve after an edit keeps the last final result shown until its own final round,
    // so the figures never step through the rounds; estimates show only where nothing final is
    // shown yet: before the first final result, after a whole-game load, or for a city just opened
    const shown = cityDemandViews.peek();
    const showEstimates = !demandView.peek()?.final || (!!this.hold && !this.cacheShown()) || cities.some((id) => !shown.has(id));
    const barrier = showEstimates || !!this.hold;
    this.cur = { version, svc, layout, results: PERIODS.map(() => null), byRound: Array.from({length: ROUNDS + 1}, () => PERIODS.map(() => null)), published: -1, arrivedAt, sentAt: performance.now(), periodMs: [], setupMs: [], memBytes: [], cities, markets: [...this.opened].filter(isMarket).map(marketZone), coarse: this.coarseOf(cities), showEstimates, barrier };
    this.log.push(this.cur);
    if (this.log.length > 50) this.log.shift();
    demandUpdating.value = true;
    // T-1790: after an edit the rounds all land at the end, so the steps are the local solve and the
    // whole one (`plan` says whether there is a local one), not rounds
    demandStep.value = { done: 0, of: barrier ? ROUNDS + 1 : 1 };
    demandScope.value = cities.length;
    // T-327: rounds in step only where clicks are answered by round (`target`): it costs the final
    // about 0.7 s on the real save
    demandSolving.value = null;
    const next = this.firstZones().slice(1);
    // T-2459: the focus pool solves the zone in view to its last round, so the city view's zone's
    // cluster goes ahead of it
    const focused = barrier && next.length > 0 && !perfOff("focusAhead") && this.focusSolves(svc, activeCity.peek());
    if (focused) this.cur.focused = true;
    const msg: ToDemand = { kind: "solve", version, net, ids, rounds: ROUNDS, barrier, first: activeCity.peek(), ...(next.length ? { next } : {}), ...(focused ? { focused } : {}), ...(perfOff("clusters") ? { whole: true } : {}), ...(perfOff("split") ? { oneMarket: true } : {}), ...(perfOff("reach") ? { wide: true } : {}), ...(perfOff("chunks") ? { exactMarket: true } : {}), ...(perfOff("periods") ? { fresh: true } : {}), ...(perfOff("estimate") ? { exact0: true } : {}), ...(this.freshParts ? { freshParts: true } : {}), ...(perfOff("noise") ? { exactKeep: true } : {}) };
    this.freshParts = false;
    this.postSolve(msg);
    if (barrier) this.askTurn();
    this.progress(); // T-243: a held game's solve shows its progress even with nothing to load
  }

  /** A solve to every worker, each solving worker told its part of the origins (T-1543). */
  private postSolve(msg: Extract<ToDemand, { kind: "solve" }>) {
    const n = this.computeN;
    this.workers.forEach((w, i) => w.postMessage(n > 1 && i < n ? { ...msg, part: [i, n] } : msg));
  }

  // --- Demand views (T-078; types and meaning in game/demand.ts) ---------------------------------

  /** T-2045: a question to the focus pool's aux worker */
  private askFocusAux(make: (id: number) => Query, zone?: CityId): Promise<Reply[]> {
    return this.ask(make, [this.computeN], [], zone, undefined, true);
  }

  /** Send `make(id)` to the given workers (all by default) and collect their replies. */
  private ask(make: (id: number) => Query, which: number[] = this.workers.slice(0, this.computeN).map((_, i) => i), transfer: Transferable[] = [], zone?: CityId, t?: Target, toFocus = false): Promise<Reply[]> {
    this.syncFocus();
    const id = this.nextQuery++;
    // `zone`: a nearby zone's question (T-276), about that zone whatever the focus does meanwhile
    const city = zone ?? activeCity.peek(), epoch = this.focusEpoch;
    const q = make(id);
    // T-2045: a question about a focus solve goes to the focus pool, the same worker index (the aux's
    // to its aux, any other to its solving workers)
    const fp = this.fpool, version = (q as { version?: number }).version;
    const pool = fp?.alive && (toFocus || (version !== undefined && this.fversions.has(version))) ? fp : null;
    if (pool) which = which.includes(this.computeN) ? [pool.auxAt] : pool.workers.slice(0, pool.n).map((_, i) => i);
    const workers = pool?.workers ?? this.workers;
    return new Promise((done) => {
      this.pendingQueries.set(id, { left: which.length, replies: [], done: replies => done(zone !== undefined || epoch === this.focusEpoch ? replies : []), focus: !!pool });
      this.qlog.push({ id, kind: q.kind, sent: performance.timeOrigin + performance.now(), replies: [] });
      if (this.qlog.length > 200) this.qlog.shift();
      // T-260: the kind of day shown; each answer weighs its periods by it
      const round = t?.round ?? demandView.peek()?.round ?? -1;
      for (const i of which) workers[i].postMessage({...q, city, round, day: shownDay.peek(), ...(t?.wait ? { wait: true } : {})}, transfer);
    });
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
    if (this.fversions.has(t.version)) return this.fshown.get(activeCity.peek())?.s.version === t.version;
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
    const t = this.target();
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
    const load = (id: CityId) => (viaFocus(id) ? Promise.resolve(true) : this.whenOpen(id)).then(async (ok): Promise<CityCells | null> => {
      if (!ok) return null;
      const [r] = viaFocus(id) ? await this.askFocusAux((q) => ({ kind: "cells", id: q, frame: frameOf() }), other) : await this.ask((q) => ({ kind: "cells", id: q, frame: frameOf() }), [this.computeN], [], other);
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
    // waits for the viewed city to open (T-149: it may be loading after a pan); a failed answer is
    // not kept, so the bubbles' next ask tries again (T-1493)
    if (!this.cellsP) {
      const p = (this.cellsP = load(activeCity.peek()));
      p.then((v) => { if (!v && this.cellsP === p) this.cellsP = null; });
    }
    return this.cellsP;
  }
  /** Nearby zones' cells (T-276), `zone:frame epoch`. */
  private otherCells = new Map<string, Promise<CityCells | null>>();
  /** A zone's cells in every frame, when it closes or its frame is out of date. */
  private dropCells(id: string) {
    for (const k of this.otherCells.keys()) if (k.slice(0, k.lastIndexOf(":")) === id) this.otherCells.delete(k);
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
          const [r] = await this.ask((id) => ({ kind: "bare", id, version, purpose: kind }), [this.computeN], [], other);
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

  /** T-2365: a zone's trips a day by length on the day shown (game/tripLengths.ts), from every
   * solving worker's periods, or for a zone open bare the aux worker's; cached per zone, solve,
   * round and day; null while not known. */
  tripLengths(zone?: CityId, ahead?: Target): Promise<TripLengths | null> {
    this.syncFocus();
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
        ? await this.ask((id) => ({ kind: "tripLengths", id, version: 0, bare: true }), [this.computeN], [], city)
        : byMask(await this.ask((id) => ({ kind: "tripLengths", id, version: at.version }), undefined, [], city, at));
      let mask = 0, sum: Float64Array | null = null, edges: Float32Array | null = null;
      for (const r of replies) {
        if (r?.kind !== "tripLengths" || !r.commute) return null;
        mask |= r.mask;
        edges = r.edges;
        sum ??= new Float64Array(r.commute.length);
        for (const v of [r.commute, r.everyday]) if (v?.length === sum.length) addInto(sum, v);
      }
      if (mask !== ALL_PERIODS || !sum || !edges || shownDay.peek() !== day) return null;
      return tripLengthsOf(version, day, edges, sum);
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
      if (this.isBare(city)) return this.bareModes(city, everydayToo, version, round, day, other);
      let mask = 0;
      let sums: Float32Array | null = null;
      let extra: Float32Array | null = null;
      // every worker sent its sums with the round (no question to a worker busy with the next
      // round); asked for only when one is missing. Another day than a weekday's only for the
      // zones on screen when the round was solved (T-1904)
      const sent = (f ? this.fpushed : this.pushed).get(`${at.version}:${at.round}:${city}:${day}`);
      const parts: { mask: number; sums: Float32Array | null; everyday: Float32Array | null }[] = sent && sent.length === (f ? this.fpool!.n : this.computeN) ? sent : (await this.ask((id) => ({ kind: "subSums", id, version: at.version }), undefined, [], other, other === undefined ? this.liveAt(at) : at)).map((r) => (r.kind === "subSums" ? r : { mask: 0, sums: null, everyday: null }));
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
      const [r] = await this.ask((id) => ({ kind: "cellModes", id, version: at.version, sums: sums! }), [this.computeN], [sums.buffer], other, at);
      if (r?.kind !== "cellModes" || !r.modes) return null;
      // (T-1533: a later round of the same solve published meanwhile still takes these: its bubbles ask
      // again, and an older round's beat none while a slow answer chases the rounds)
      if (!stillShown(other, city, version, day)) return null;
      // Bubbles now count trip legs at each home/activity pair, both directions.
      // Old commuter queries count one person; convert before adding everyday travel.
      // (work alone for the purpose filter, T-158: `everydayToo` false)
      for (let k = 0; k < r.modes.length; k++) r.modes[k] = 2 * r.modes[k] + (everydayToo ? extra?.[k] ?? 0 : 0);
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
  private async bareModes(city: CityId, everydayToo: boolean, version: number, round: number, day: DemandDay, other: CityId | undefined): Promise<CellModes | null> {
    const [r] = await this.ask((id) => ({ kind: "bare", id, version, purpose: everydayToo ? -1 : -2 }), [this.computeN], [], other);
    if (r?.kind !== "bare" || !r.commute || !r.commute.length) return null;
    let extra: Float32Array | null = null;
    for (const p of bareParts(r)) {
      if (!extra) extra = p.slice();
      else addInto(extra, p);
    }
    if (!stillShown(other, city, version, day) || demandView.peek()?.round !== round) return null;
    const m = r.commute;
    for (let k = 0; k < m.length; k++) m[k] = 2 * m[k] + (everydayToo ? extra?.[k] ?? 0 : 0);
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
    for (const { o, n, stops } of solve.layout) for (let k = 0; k < n; k++) if (inside.has(stops[k])) nodes.push(o + k, o + 2 * n - 1 - k);
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
  private async transferParts(t: Target, nodes: number[]): Promise<Float32Array[] | null> {
    const ids = Uint32Array.from(nodes), parts: Float32Array[] = [];
    let mask = 0;
    for (const r of byMask(await this.ask((id) => ({ kind: "transfers", id, version: t.version, nodes: ids }), undefined, [], undefined, t))) {
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
    for (const [key, riders] of sum) {
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
    for (const r of await this.ask((id) => ({ kind: "trip", id, version: t.version, from, to, ...(purpose >= 0 ? { purpose } : {}) }), bare ? [this.computeN] : undefined, [], undefined, t)) {
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
    const sq = !near && this.isBare(activeCity.peek()) ? await this.bareSquares(t.view, e, cells) : undefined;
    if (sq === null) return null;
    const at = routeNodes(solve.layout);
    const periods: (MarketPeriod | null)[] = PERIODS.map(() => null);
    let mask = 0;
    for (const r of await this.ask((id) => ({ kind: "icTrip", id, version: t.version, end: e, cells, far, farGid, ...(sq ? { sq } : {}), ...(near ? { near } : {}) }), undefined, [], undefined, t)) {
      if (r.kind !== "icTrip" || !r.answers) return null;
      mask |= r.mask;
      r.periods.forEach((q, i) => (periods[q] = readMarketPeriod(r.answers![i], at)));
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
    const p = this.viaRound(`flows|${end}|${cells.join(",")}`, (t) => this.flowsAt(t, end, cells));
    this.flowsPending = p;
    const done = () => { if (this.flowsPending === p) this.flowsPending = null; };
    p.then(done, done);
    return p;
  }
  private async flowsAt(t: Target, end: "home" | "work", cells: Uint32Array): Promise<Flows | null> {
    const city = activeCity.peek(), round = demandView.peek()?.round;
    const version = t.view;
    const e = end === "home" ? 0 : 1;
    // the commutes' far end, packed as `flows` answers, and each worker's `purposes` answer
    let p: Float32Array;
    let purposeParts: PurposePart[];
    if (this.isBare(city)) {
      // T-336: a zone open bare: the aux worker's no-rail pass, one part per solving worker's periods
      const [r] = await this.ask((id) => ({ kind: "bareFlows", id, version, end: e, cells }), [this.computeN]);
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
      const [r] = await this.ask((id) => ({ kind: "flows", id, version: t.version, end: e, cells, sums: sums! }), [0], [sums.buffer]);
      if (r?.kind !== "flows" || !r.packed) return null;
      p = r.packed;
      purposeParts = [];
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
  private sumsCtx(own = -1): SumsCtx {
    return { market: MARKET_SPEC, world: frameOf(), places: INTERCITY_PLACES.length, own, modes: ownModes(this.modesTable, own), centres: centresOfPlaces() };
  }

  /** T-336: a selection of the zone in view, open bare, as the regional market's squares, from the
   * aux worker (the solving workers hold the market but not the zone); null when it has none. */
  private async bareSquares(version: number, end: number, cells: Uint32Array): Promise<{ xy: Float32Array; share: Float32Array } | null> {
    const [r] = await this.ask((id) => ({ kind: "bareSquares", id, version, end, cells }), [this.computeN]);
    return r?.kind === "bareSquares" && r.xy && r.share ? { xy: r.xy, share: r.share } : null;
  }

  /** The longer trips of a bubble selection (game/demand.ts `IntercitySelection`, T-118). */
  intercitySelection(end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null> {
    if (cells.length === 0) return Promise.resolve(null);
    return this.viaRound(`icSel|${end}|${cells.join(",")}`, (t) => this.intercitySelectionAt(t, end, cells));
  }
  private async intercitySelectionAt(t: Target, end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null> {
    const version = t.view;
    const e = end === "home" ? 0 : 1;
    const sq = this.isBare(activeCity.peek()) ? await this.bareSquares(version, e, cells) : undefined;
    if (sq === null) return null;
    if (this.flowsPending) await Promise.race([this.flowsPending, new Promise((r) => setTimeout(r, FLOWS_FIRST_MS))]);
    const parts = await this.partsIntercity("icSel", t, (id) => ({ kind: "icSel", id, version: t.version, end: e, cells, ...(sq ? { sq } : {}) }));
    if (!parts) return null;
    const own = CITIES.find((c) => c.id === activeCity.peek())?.place ?? -1;
    const ctx = this.sumsCtx(own);
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
    const ctx = this.sumsCtx();
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
  private async daysOf(s: Solve, layout: Solve["layout"], round: number, final: boolean, res: Result[], noCorridors = false): Promise<DayDemand[]> {
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
        if (d) { d.stops = l.stops; if (sl) { d.tph = sl.tph; d.cars = sl.cars; } }
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
    const shown = demandByDay.peek()[0].view;
    const s = shown?.final ? this.log.find((x) => x.version === shown.version) : undefined;
    const res = s?.results;
    if (!s || !shown || !res?.length || res.some((r) => !r)) return;
    this.madeKey = key;
    const round = shown.round;
    this.inOrder(async () => {
      if (demandByDay.peek()[0].view !== shown) return;
      demandByDay.value = await this.daysOf(s, s.layout, round, true, res as Result[]);
    });
  }

  /**
   * T-1008: a kind of day's corridors with the market's table (sent once, shared by every result);
   * none while the table is lost (a newer market's pushed it out), which the workers then send
   * again with their next results.
   */
  private corridorsOf(c: CorridorRail | null): Corridors {
    if (!c) return NO_CORRIDORS;
    const table = this.corridorTables.get(c.key);
    if (!table) {
      if (!this.tableAsked.has(c.key)) { this.tableAsked.add(c.key); for (const w of this.workers) w.postMessage({ kind: "corridorsAgain" } satisfies ToDemand); }
      return NO_CORRIDORS;
    }
    this.corridorTables.delete(c.key);
    this.corridorTables.set(c.key, table);
    return { table, longerScale: c.longerScale, commuteScale: c.commuteScale, railAt: c.railAt, longerRail: c.longerRail, commuteRail: c.commuteRail };
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
      // T-324: lines new or changed since the shown result have no figures; theirs come from the
      // first estimate at once, every other figure waits for the final round as above
      if (round === 0 && s === this.cur) this.publishNewLines(s, res);
      return;
    }
    this.inOrder(async () => {
      const days = await this.daysOf(s, s.layout, round, final, res);
      // a cache of the same network shown meanwhile (T-328) holds its estimates back
      if (!final && !s.showEstimates) return;
      // T-2045: the zones the focus pool has as far on keep its figures until this solve's final
      demandByDay.value = final ? days : this.withFocus(days, s, round);
      this.lodShown(s.cities, s.coarse);
      if (final && s === this.cur) this.endFocus();
      else this.retireFocus();
      if (final) {
        s.refinedAt = performance.now();
        const k = this.uncovered.findIndex((u) => u.svc === s.svc);
        if (k >= 0) {
          this.uncovered.splice(0, k + 1);
          demandBehindFrom.value = this.uncovered[0]?.minute ?? null;
        }
        // an older solve's figures are never read again (its svc and layout are); a few MB each
        for (const o of this.log) if (o.version < s.version) (o.results = []), (o.byRound = []), (o.clusterRes = undefined);
        this.aheadP.clear();
        this.cityVersion.clear();
        if (s === this.cur) {
          demandUpdating.value = false;
          demandSolving.value = null;
          this.unsettled = null;
          setTimeout(() => void this.writeCache(s), 0); // T-328; a task of its own (T-789)
          // T-1734: what was held back meanwhile
          if (s.barrier === false) this.flush();
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
  private publishNewLines(s: Solve, res: Result[]) {
    this.inOrder(async () => {
      const shown = demandByDay.peek()[0].view;
      const base = this.log.find((x) => x.version === shown?.version);
      if (!base || !shown) return;
      const had = new Map(base.layout.map((l) => [l.id, l.stops.join(",")]));
      // a line given early figures by an earlier solve's estimate is still "new" against the base
      const fresh = s.layout.filter((l) => had.get(l.id) !== l.stops.join(","));
      if (!fresh.length) return;
      s.earlyAt = performance.now();
      s.earlyLines = fresh.map((l) => l.id);
      const ests = await this.daysOf(s, fresh, 0, false, res, true);
      demandByDay.value = demandByDay.peek().map((d, k) => {
        if (!d.view) return d;
        const est = ests[k].view!;
        const lines = new Map(d.view.lines), stations = new Map(d.view.stations);
        for (const { id } of fresh) {
          const l = est.lines.get(id);
          if (l) lines.set(id, { ...l, estimate: true });
        }
        for (const [id, v] of est.stations) if (!stations.has(id)) stations.set(id, v);
        let purposes = d.view.purposes;
        if (est.purposes) {
          purposes = new Map(purposes);
          for (const [key, p] of est.purposes) {
            const old = purposes.get(key);
            const pl = new Map(old?.lines), ps = new Map(old?.stations);
            for (const { id } of fresh) { const l = p.lines.get(id); if (l) pl.set(id, l); }
            for (const [id, v] of p.stations) if (!ps.has(id) && !d.view.stations.has(id)) ps.set(id, v);
            purposes.set(key, { lines: pl, stations: ps });
          }
        }
        return { ...d, view: { ...d.view, lines, stations, purposes } };
      });
    });
  }

  /** T-1733: the whole solve after an edit's local results may start (once a version) */
  private wholeGo(s: Solve) {
    if (s !== this.cur || s.wholeSent) return;
    s.wholeSent = true;
    for (const w of this.workers.slice(0, this.computeN)) w.postMessage({ kind: "whole", version: s.version } satisfies ToDemand);
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
    demandByDay.value = demandByDay.peek().map((d, k) => {
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
  private publishCluster(s: Solve, res: Result[], c: { cities: CityId[]; market: boolean; lines: Uint32Array; local?: boolean }, round: number) {
    if (s !== this.cur) return;
    const final = round >= ROUNDS;
    // T-1533: after an edit, the zone in view's cluster's free-flow round gives new lines their figures
    if (round === 0 && !final && !s.showEstimates) return this.publishNewLines(s, res);
    // T-1790: the zone in view's cluster solved ahead counts its own rounds
    if (s.barrier && c.cities.includes(activeCity.peek())) demandStep.value = { done: round + 1, of: ROUNDS + 1 };
    // T-2453: its zones' questions now go to its next round
    if (s.barrier && !final) this.askTurn();
    const fresh = Array.from(c.lines, (l) => s.layout[l]).filter(Boolean);
    this.inOrder(async () => {
      const shown = demandByDay.peek()[0].view;
      // T-402: with nothing shown yet, or only this solve's own figures of the cluster, a view of the
      // cluster alone, which the solve's own results replace (its totals are the cluster's, so never
      // the solve's final); its rounds before the last only where the solve's estimates are shown
      const alone = !shown || shown.version === s.version;
      if (!final && s.showEstimates && !alone) return this.selectedRound(s, res, c.cities, round);
      if (!final && (!s.showEstimates || !alone)) return;
      if (alone) demandByDay.value = await this.daysOf(s, fresh, round, false, res);
      else {
        const ests = await this.daysOf(s, fresh, round, final, res, !c.market);
        const ids = new Set(s.layout.map((l) => l.id));
        demandByDay.value = demandByDay.peek().map((d, k) => {
          if (!d.view) return d;
          const est = ests[k];
          const lines = new Map(d.view.lines), stations = new Map(d.view.stations);
          // lines the network no longer has go too
          for (const id of lines.keys()) if (!ids.has(id)) lines.delete(id);
          for (const [id, l] of est.view!.lines) lines.set(id, l);
          for (const [id, v] of est.view!.stations) stations.set(id, v);
          let purposes = d.view.purposes;
          if (est.view!.purposes) {
            purposes = new Map(purposes);
            for (const [key, p] of est.view!.purposes) {
              const old = purposes.get(key);
              const pl = new Map(old?.lines), ps = new Map(old?.stations);
              for (const [id, l] of p.lines) pl.set(id, l);
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
      }
      this.lodShown(c.cities, s.coarse);
      if (final) {
        for (const id of c.cities) this.cityVersion.set(id, s.version);
        if (s.barrier) this.askTurn(); // T-2453: its zones' questions now go to its final
        // T-2045: a cluster's own final beats the focus pool's figures of its zones
        if (c.cities.some((id) => this.fshown.has(id))) {
          const z = new Map(focusZones.peek());
          for (const id of c.cities) { this.fshown.delete(id); z.delete(id); }
          focusZones.value = z;
        }
        // T-1733: its lines' riders are worked out (a local result's too, longer trips as last solved)
        linesSettled(fresh.map((l) => l.id));
        if (c.local) {
          s.localShownAt ??= performance.now();
          setTimeout(() => this.wholeGo(s), EDIT_SETTLE_MS);
        }
        // T-1790: an edit's step 2 (the whole solve) once every local result is in
        if (c.local && s === this.cur && (s.localShown = (s.localShown ?? 0) + 1) >= Math.max(...(s.localPlans ?? [0])) && demandStep.peek().of === 2) {
          demandStep.value = { done: 1, of: 2 };
          demandScope.value = s.cities.length;
        }
        const left = new Set(demandSolving.peek() ?? []);
        for (const id of c.cities) { left.delete(id); this.unsettled?.delete(id); }
        demandSolving.value = left;
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
    const days = demandByDay.peek();
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
    if (changed) demandByDay.value = next;
  }


}

/** The days the demand views are made for (T-1373): every day, the going-out and coming-back days
 * only while one is near. */
function madeDays(): DemandDay[] {
  return DEMAND_DAYS.filter((k) => (k !== OUT_DAY && k !== BACK_DAY) || rushRun.peek() !== null);
}

let client: DemandClient | null = null;

const TEST = new URLSearchParams(location.search).get("demandTest") === "1";
const DEBUG_LOD = new URLSearchParams(location.search).get("debug") === "1";
/** T-351: the page starts a new game (`?new=1`, or the start menu's New game, T-677): no stations,
 * so the viewed zone opens bare from the start. */
let NEW_GAME = false;

/** Start the pool for a city. With `?demandTest=1` it solves T-005's hand-made network. */
export function startDemand(city = HOME_CITY, newGame = false) {
  if (client) return client;
  NEW_GAME = newGame && !TEST;
  const c = (client = new DemandClient(city));
  onLoadStop((why) => c.fail(why));
  setTripQuery((from, to, purpose) => c.trip(from, to, purpose)); // T-195
  setMarketTripQuery((end, cells, far, farGid, near) => c.marketTrip(end, cells, far, farGid, near)); // T-866, T-885
  setDemandQueries({ transfers: (members) => c.transfers(members), platformTransfers: () => c.platformTransfers(), cityCells: (city) => c.cityCells(city), cellModes: (purpose, city) => c.purposeModes(purpose ?? "all", city), stationRiders: (id, top, members) => c.stationRiders(id, top, members), flows: (end, cells) => c.flows(end, cells), intercitySelection: (end, cells) => c.intercitySelection(end, cells), intercityStation: (id, members) => c.intercityStation(id, members), tripLengths: (zone) => c.tripLengths(zone) });
  if (TEST) c.submitNetwork(demandTestSnapshot());
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twDemand = { client: c, log: c.log, days: () => demandByDay.peek(), start: c.start, resubmit: () => c.resubmit(), view: () => demandView.peek(), cities: () => cityDemandViews.peek(), nearby: () => nearbyZones.peek(), intercity: () => intercityDemand.peek(), activeCity, cellModes: () => c.cellModes(), stationRiders: (id: string) => c.stationRiders(id), cityCells: () => c.cityCells(), flows: (end: "home" | "work", cells: Uint32Array) => c.flows(end, cells), intercitySelection: (end: "home" | "work", cells: Uint32Array) => c.intercitySelection(end, cells), intercityStation: (id: string) => c.intercityStation(id), transfers: (members: string[]) => c.transfers(members), trip: (from: [number, number], to: [number, number], purpose?: number) => c.trip(from, to, purpose), tripLengths: (zone?: CityId) => c.tripLengths(zone) };
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
