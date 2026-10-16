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
import { activeCity, CITIES, cityById, getWorldFrame, HOME_CITY, LEGACY_FRAME, marketToWorld, viewedZone, worldFrameEpoch, type CityId } from "../game/cities";
import type { FrameSpec } from "../game/frame";
import { DEMAND_DAYS, type DemandDay } from "../game/days";
import { display, shownDay, tool } from "../game/ui";
import {
  cityDemandViews, demandByDay, intercityDemand,
  type DayDemand,
  type DemandView,
  demandStep,
  demandUpdating,
  demandView,
  setDemandQueries,
  type CellModes,
  type CellRiders,
  type CityCells,
  type ComplexTransfers,
  type TransferPair,
  type FarEnd,
  type Flows,
  type PurposeDemand,
  type IntercitySelection,
  type IntercityStation,
  type PlaceFlows,
  type PlaceTrips,
  type ZoneTrips,
  INTERCITY_PLACES,
  type LineDemand,
  type StationDemand,
  type StationRiders,
  type ZoneRiders,
  demandSolving,
} from "../game/demand";
import { makeFrame } from "../game/frame";
import { demandIdle, loadComplete, loadFailed, loadProgress, startup, takeGameLoadHold, zoneLoads } from "../game/loading";
import { PERIODS, periodsMatch } from "../game/periods";
import { EVERYDAY_KEYS, membersOf, PURPOSES, shownKey, type PurposeFilter, type PurposeKey } from "../game/purposes";
import { perfOff } from "../perfFlags";
import { marketZonesNear } from "./demandClusters";
import { citiesReached, footprintOf, loadFootprints, REACH_M, zonesNearStations, type Footprint } from "./cityReach";
import { nearbyZones } from "../game/zones";
import type { BareFigures, DemandNetwork, FromDemand, ToDemand } from "./demandProtocol";
import { readTripPeriod, setTripQuery, tripKey, type RouteNodes, type Trip } from "../game/trip";
import { demandTestSnapshot } from "./demandTest";
import { mergeEveryday } from "./selectionMerge";
import type { NetworkSnapshot } from "./protocol";
import { saveSlotsState } from "./clockClient";
import { readDemandCache, writeDemandCache } from "./saveFile";
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

/**
 * Workers in the pool (T-361, T-381). By threads: 12 or more get 3, 6 to 11 get 2, fewer 1; then
 * by memory (each worker's WASM is a few hundred MB, notes/T-361.md): 4 GB or less (`navigator.deviceMemory`, Chrome
 * and Edge only) gets 1, and where it is not reported (Firefox, Safari) at most 2. The "Use more
 * cores" setting restores the older rule, half the threads less one, 1 to 5 (one per period).
 * `?demandWorkers=N` (1-5) overrides everything. The pool is built once, so the setting counts at
 * the next load.
 */
/**
 * The periods solving worker `i` of `n` owns: by index (`q % n`), unless that gives one worker both
 * periods of a demand level (2 workers: both peaks), when they are dealt out level by level. An edit
 * to one level's trains re-solves only that level's periods (T-124), so they go to different workers.
 */
export function periodsOfWorker(i: number, n: number): number[] {
  const qs = PERIODS.map((_, q) => q);
  const clash = qs.some((q) => qs.some((s) => s > q && PERIODS[s].demand === PERIODS[q].demand && s % n === q % n));
  const LEVEL = { high: 0, medium: 1, low: 2 };
  const order = clash && !perfOff("spread") ? qs.slice().sort((a, b) => LEVEL[PERIODS[a].demand] - LEVEL[PERIODS[b].demand] || a - b) : qs;
  return order.filter((_, k) => k % n === i).sort((a, b) => a - b);
}

export function poolSize(): number {
  const q = Number(new URLSearchParams(location.search).get("demandWorkers"));
  if (Number.isInteger(q) && q >= 1 && q <= PERIODS.length) return q;
  const hc = navigator.hardwareConcurrency || 4;
  if (display.peek().moreCores) return Math.max(1, Math.min(PERIODS.length, Math.floor(hc / 2) - 1));
  const byThreads = hc >= 12 ? 3 : hc >= 6 ? 2 : 1;
  const mem = (navigator as { deviceMemory?: number }).deviceMemory;
  if (mem === undefined) return Math.min(byThreads, 2);
  return mem <= 4 ? 1 : byThreads;
}

/** The service demand reads, with the UI's ids. */
interface Service {
  frame: FrameSpec;
  fare: { base: number; perKm: number };
  /** `complex`: the station's complex (T-161); absent = a complex of its own */
  stations: { id: string; x: number; y: number; complex?: number }[];
  lines: { id: string; stops: string[]; tph: [number, number, number]; hopS: number[][][]; hopKm?: number[][]; dwellS: number[][][]; cars: number }[];
}

type Result = Extract<FromDemand, { kind: "result" }>;
type QueryKind = "cells" | "subSums" | "cellModes" | "station" | "flowRail" | "flows" | "purposes" | "icSel" | "icStation" | "purposeModes" | "transfers" | "trip" | "bare" | "bareFlows" | "bareSquares";
type Reply = Extract<FromDemand, { kind: QueryKind }>;
type QueryMessage = Extract<ToDemand, { kind: QueryKind }>;
type WithoutCity<T> = T extends unknown ? Omit<T, "city" | "round"> : never;
type Query = WithoutCity<QueryMessage>;
/** T-327: where a view's question goes: the solve and round the workers are asked about, the shown
 * result's version the answer is given for, and whether a worker short of the round waits for it. */
/** `shown` (T-402): the round published of the rounds `round` counts (the cluster's solved ahead,
 * or the solve's), when not the shown view's */
interface Target { version: number; round: number; view: number; wait: boolean; solve?: Solve; shown?: number }
const ALL_PERIODS = (1 << PERIODS.length) - 1;
const frameOf = (): FrameSpec => { const f = getWorldFrame(); return { kind: f.kind, lon: f.lon, lat: f.lat }; };
/** A point the regional market reports (in its own pack frame) as a world point. */
const marketXY = (x: number, y: number): { x: number; y: number } => marketToWorld(x, y);
/** A place's centre (lon, lat) in world metres. */
const centreXY = ([lon, lat]: [number, number]): { x: number; y: number } => { const [x, y] = getWorldFrame().fromLonLat(lon, lat); return { x, y }; };
/** The zone count of a longer-trip answer of `len` numbers. For M markets a selection has 1+8*M
 * header numbers, 4 per zone of flows, 3 per zone of x, y, market, then (T-276) the commutes
 * between zones: 4 per market and 4 per zone, then (T-378) the trips with places that have no
 * zone, starting and ending; a station has 6*M header numbers then 5 per zone. */
function intercityZones(kind: "icSel" | "icStation", len: number): number {
  const m = INTERCITY_PLACES.length;
  return kind === "icSel" ? (len - 3 - 12 * m) / 11 : (len - 6 * m) / 5;
}
/** Where the zones' own x, y, market start: the same on every worker, so never added up. */
function intercityZoneXY(kind: "icSel" | "icStation", len: number): number {
  const nz = intercityZones(kind, len);
  return kind === "icSel" ? 1 + 8 * INTERCITY_PLACES.length + 4 * nz : 6 * INTERCITY_PLACES.length + 2 * nz;
}
/** Add a worker's longer-trip answer into `sums`: everything but the matched share (a selection's
 * first number) and the zones' x, y, market. */
function addIntercity(kind: "icSel" | "icStation", sums: Float64Array, v: Float64Array) {
  const xy = intercityZoneXY(kind, sums.length), end = xy + 3 * intercityZones(kind, sums.length);
  for (let k = kind === "icSel" ? 1 : 0; k < sums.length; k++) if (k < xy || k >= end) sums[k] += v[k];
}

/** T-378: the index of the corridor between places `a` and `b` (`IntercityApi.corridor_summary`
 * order: (0, 0), (0, 1) .. (1, 1) ..). */
function corridorOf(a: number, b: number): number {
  const n = INTERCITY_PLACES.length, i = Math.min(a, b), j = Math.max(a, b);
  return i * n - (i * (i - 1)) / 2 + (j - i);
}
/** T-378: a corridor's longer trips not by rail as [by plane, by car] (its shares from
 * `IntercityApi.corridor_modes`; before any are known, all by car). Near places on two landmasses
 * have neither. */
function corridorSplit(modes: Float32Array | null, c: number, rest: number): [number, number] {
  if (!modes || 2 * c + 1 >= modes.length) return [0, rest];
  return [rest * modes[2 * c], rest * modes[2 * c + 1]];
}

/** A fingerprint of an answer's exact numbers: stations the demand model merged answer with the
 * same numbers, so a station complex counts them once (T-146). */
function signature(a: Float32Array | Float64Array): string {
  const bits = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4);
  let h = 0x811c9dc5;
  for (let i = 0; i < bits.length; i++) h = Math.imul(h ^ bits[i], 0x01000193);
  return `${a.length}:${h >>> 0}`;
}

/** Workers' replies in the order of the periods they own (each worker's set is fixed), so sums and
 * fingerprints do not depend on which worker answered first. */
function byMask(parts: Reply[]): Reply[] {
  const m = (r: Reply) => ("mask" in r ? r.mask : 0);
  return [...parts].sort((a, b) => m(a) - m(b));
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
  /** T-232: its estimates before the final round are shown (startup, a game load, a city with no
   * figures yet); otherwise the last final result stays shown until this one's final round */
  showEstimates: boolean;
  /** T-326: the last worker's clusters kept and solved */
  clusters?: [number, number];
  /** T-124: per solving worker, its periods kept and solved in the clusters it solved */
  periods?: [number, number][];
  /** T-392: each cluster published ahead of the solve's final, by its cities: its results per
   * round and period as they come in, the last round in from every worker, and when its final
   * round was shown */
  clusterRes?: Map<string, { byRound: (Result | null)[][]; published: number; at?: number }>;
  /** T-402: the cluster in view, stepped through its rounds alone ahead of the rest (a solve with
   * the barrier, `plan.ahead`): its cities and key in `clusterRes` */
  ahead?: { cities: CityId[]; key: string };
}

/** When one item (a city or the market) was asked for, downloaded and open in every worker:
 * performance.now() (window.twDemand.start.loads with ?debug=1). */
interface LoadLog { item: string; requestedAt: number; downloadedAt?: number; openedAt?: number; bare?: boolean; closedAt?: number }
/** One file being downloaded, for the loading percentage (T-149). */
interface Download { got: number; total: number; at: number }
/**
 * T-243: the loading bar moves in proportion to time. Each step (downloading, opening in every
 * worker, solving every period to its first estimate) weighs as long as it is expected to take,
 * and within a step the bar moves with the time spent against that expectation, up to
 * STEP_CAP, until the step's real completion lands. Expected times, measured on Anita's desktop
 * (notes/T-243.md): opening a city pack 130 ms per MB of its .bin (New York's 16 MB: 2.1 s),
 * the regional market 40 ms per MB; the first estimate 600 ms plus 2.2 ms per line stop (an empty
 * network 0.6 s, the real New York save's 1,756 stops 4.4 s). A download's expected length is its
 * own rate so far. The last startup's solve time and stop count are remembered per browser
 * (LAST_SOLVE) and scale the estimate, so a slower machine corrects it after one load; until the
 * clock worker has sent the network, its stop count stands in (a reload opens the same game).
 * The bar never goes back: when an expectation grows, the rest of the work is spread over the
 * rest of the bar.
 */
const OPEN_MS_PER_MB = { city: 130, market: 60 };
const SOLVE_MS = { base: 600, perStop: 2.2 };
const STEP_CAP = 0.98;
/** a pack whose size is not known yet (its download has not started): about a mid-sized city */
const GUESS_BYTES = 6e6;
const LAST_SOLVE = "anitabuilder-load-solve";
/** T-377: a zone's market pack as a load item, "market:<zone>" */
const MARKET = "market:";
const marketItem = (id: CityId) => MARKET + id;
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
/** T-347: until a load has timed its final round, the final is guessed at this many first estimates */
const FINAL_FACTOR = 2.5;
/** a share of `expected` ms gone since `since`: in step to 90%, then slowing toward STEP_CAP, so a
 * step that runs late still creeps rather than stopping dead */
function timeShare(since: number | undefined, expected: number, now: number): number {
  if (since === undefined) return 0;
  const x = Math.max(0, now - since) / Math.max(1, expected);
  return x < 0.9 ? x : 0.9 + (STEP_CAP - 0.9) * (1 - Math.exp(-(x - 0.9) / 0.3));
}
/** `finalMs`: to the card going (T-421: the zone in view's final); `allMs`: to the solve's final */
function lastSolve(): { ms: number; stops: number; finalMs?: number; allMs?: number } | null {
  try {
    const v = JSON.parse(localStorage.getItem(LAST_SOLVE) || "null");
    return v && v.ms > 0 && v.stops >= 0 ? v : null;
  } catch { return null; }
}
function rememberSolve(ms: number, stops: number, finalMs?: number, allMs?: number) {
  try { localStorage.setItem(LAST_SOLVE, JSON.stringify({ ms, stops, ...(finalMs ? { finalMs } : {}), ...(allMs ? { allMs } : {}) })); } catch { /* storage unavailable */ }
}
/** Expected ms from a solve's start to its first estimate, or (`toFinal`, T-347) to the card going,
 * or ("all") to the solve's final; `stops` null = the network is not known yet. */
function solveEstimate(stops: number | null, toFinal: boolean | "all" = false): number {
  const last = lastSolve();
  const n = stops ?? last?.stops ?? 0;
  const final = toFinal === "all" ? last?.allMs ?? last?.finalMs : last?.finalMs;
  const ms = last ? (toFinal ? final ?? last.ms * FINAL_FACTOR : last.ms) * solveModel(n) / solveModel(last.stops) : solveModel(n) * (toFinal ? FINAL_FACTOR : 1);
  return ms;
}

/**
 * The lines' and stations' riders by trip purpose (T-158), from every period's `parts`, laid out
 * as `publish` lays out the totals. A purpose no period reports is left out. Keyed by the shown
 * purposes: school counts university too (T-192, `membersOf`).
 */
function purposeDemand(layout: Solve["layout"], res: Result[]): Map<PurposeKey, PurposeDemand> {
  const out = new Map<PurposeKey, PurposeDemand>();
  for (const { key } of PURPOSES) {
    const members = membersOf(key);
    if (!res.some((r) => members.some((m) => r.parts?.keys.includes(m)))) continue;
    const lines: PurposeDemand["lines"] = new Map(), stations: PurposeDemand["stations"] = new Map();
    for (const { id, o, n, stops } of layout) {
      const boardings = stops.map(() => 0);
      const loads: Float32Array[] = [];
      res.forEach((r) => {
        const seg = new Float32Array(2 * (n - 1));
        loads.push(seg);
        const p = r.parts;
        if (!p) return;
        for (const m of members) {
          const at = p.keys.indexOf(m);
          if (at < 0) continue;
          const nr = p.seg.length / p.keys.length, base = at * nr;
          for (let k = 0; k < n; k++) {
            const a = base + o + k, b = base + o + 2 * n - 1 - k;
            boardings[k] += p.board[a] + p.board[b];
            let sd = stations.get(stops[k]);
            if (!sd) stations.set(stops[k], (sd = { boardings: 0, alightings: 0 }));
            sd.boardings += p.board[a] + p.board[b];
            sd.alightings += p.alight[a] + p.alight[b];
          }
          for (let k = 0; k < n - 1; k++) {
            seg[k] += p.seg[base + o + k];
            seg[n - 1 + k] += p.seg[base + o + n + k];
          }
        }
      });
      lines.set(id, { ridersPerDay: boardings.reduce((x, y) => x + y, 0), boardings, loads });
    }
    out.set(key, { lines, stations });
  }
  return out;
}

function serviceFromSnapshot(s: NetworkSnapshot): Service {
  return {
    frame: s.frame ?? LEGACY_FRAME,
    fare: s.fare ?? { base: 2, perKm: 0 }, // legacy fixed 6 perceived minutes
    stations: s.stations.map((st) => ({ id: String(st.id), x: st.x, y: st.y, complex: st.complex })),
    lines: s.lines.map((l) => ({ id: String(l.id), stops: l.stops.map(String), tph: l.tph, hopS: l.hopS, hopKm: l.hopKm, dwellS: l.dwellS, cars: l.cars })),
  };
}

/** Pack a service for the workers; lines naming a station that is not in the list are left out. */
function packService(svc: Service): { net: DemandNetwork; layout: Solve["layout"] } {
  const index = new Map(svc.stations.map((s, i) => [s.id, i]));
  const lines = svc.lines.filter((l) => l.stops.length >= 2 && l.stops.every((s) => index.has(s)));
  const nStops = lines.reduce((a, l) => a + l.stops.length, 0);
  const net: DemandNetwork = {
    frame: svc.frame,
    stXY: new Float32Array(svc.stations.flatMap((s) => [s.x, s.y])),
    // a station without a complex number is one of its own: numbered past every complex
    complex: new Uint32Array(svc.stations.map((s, i) => s.complex ?? svc.stations.length + i)),
    lineN: new Uint32Array(lines.map((l) => l.stops.length)),
    stops: new Uint32Array(nStops),
    times: new Float32Array(6 * nStops),
    tph: new Float32Array(lines.flatMap((l) => l.tph)),
    cars: new Float32Array(lines.map((l) => l.cars)),
    fare: svc.fare,
    hopKm: new Float32Array(2 * nStops),
  };
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
    }
    // Fare distances use actual track length. Old/test snapshots fall back to station distance.
    for (let run = 0; run < 2; run++) for (let k = 0; k < n - 1; k++) {
      const i = run === 0 ? k : n - 1 - k, j = run === 0 ? k + 1 : n - 2 - k;
      const from = svc.stations[index.get(l.stops[i])!], to = svc.stations[index.get(l.stops[j])!];
      net.hopKm[2 * a + run * n + k] = l.hopKm?.[run]?.[k] ?? Math.hypot(to.x - from.x, to.y - from.y) / 1000;
    }
    a += n;
  }
  return { net, layout };
}

class DemandClient {
  /** The pool's workers that solve periods, then one more with none (`aux`, demand.worker.ts). */
  private workers: Worker[] = [];
  private computeN = 0;
  private periodsOf: number[][] = [];
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
  /** T-378: the market's per-corridor shares of the longer trips not by rail by plane and by car */
  private corridorModes: Float32Array | null = null;
  /** a service waiting for the pool to start */
  private waiting: { svc: Service; arrivedAt: number } | null = null;
  private lastKey = "";
  readonly log: Solve[] = [];
  private nextQuery = 1;
  private pendingQueries = new Map<number, { left: number; replies: Reply[]; done: (r: Reply[]) => void }>();
  /** T-285: the last queries' timings (epoch ms): sent, and per reply its worker, when the worker
   * took it up, how long it took, when it landed here (window.twDemand.client.qlog, ?debug=1) */
  qlog: { id: number; kind: string; sent: number; replies: { w: number; start: number; ms: number; at: number }[] }[] = [];

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
  // --- what is loaded (T-149) ---
  /** city packs and the market ("market") being fetched or fetched */
  private fetching = new Set<string>();
  private downloads = new Map<string, Download>();
  /** workers that have opened each item, and the items every worker has opened */
  private acks = new Map<string, number>();
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
  /** each network's stations as lon, lat pairs, the zones whose boxes they come near (T-404), and
   * the zones whose market riders reach any (T-396, T-377) */
  private nearZones = new WeakMap<Service, { ll: Float64Array; near: CityId[]; markets: CityId[] }>();
  /** footprints being fetched for the reach test */
  private fetchingFootprints = new Set<CityId>();
  private onOpen: { item: string; f: (ok: boolean) => void }[] = [];
  /** the load the percentage describes: its items, the solve that completes it, its best percentage */
  private job: { items: string[]; cities: CityId[]; startup: boolean; version?: number; best: number; anchor?: { shown: number; frac: number } } | null = null;
  private progressTimer = 0;
  /** T-243: while a load runs the bar also moves with time, so progress is recomputed this often */
  private progressTick = 0;
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
  cacheLog: { at: number; hash?: string; readMs?: number; miss?: string; shownAt?: number; wrote?: { at: number; ms: number; bytes: number; ok: boolean } } | null = null;

  constructor(_city: string) {
    effect(() => { activeCity.value; worldFrameEpoch.value; this.syncFocus(); });
    const n = (this.computeN = poolSize());
    if (new URLSearchParams(location.search).has("debug")) console.info(`demand workers: ${n} (threads ${navigator.hardwareConcurrency}, memory ${(navigator as { deviceMemory?: number }).deviceMemory ?? "not reported"})`);
    for (let i = 0; i <= n; i++) {
      const w = new Worker(new URL("./demand.worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<FromDemand>) => this.receive(i, e.data);
      w.onerror = (e) => this.fail(`worker ${i}: ${e.message}`);
      this.workers.push(w);
      this.periodsOf.push(i < n ? periodsOfWorker(i, n) : []);
      // the WASM starts at once; packs follow as they arrive (T-149)
      // T-325: the aux worker is told every solving worker's periods, for the zones it holds bare
      const msg: ToDemand = { kind: "init", packs: [], periods: this.periodsOf[i], aux: i === n, warmEveryday: i === 0 && n > 1 && !perfOff("stagger"), ...(i === n ? { groups: this.periodsOf.slice(0, n) } : {}) };
      w.postMessage(msg);
    }
    // the viewed city loads when it changes (a pan, the city buttons)
    effect(() => { activeCity.value; viewedZone.value; this.ensure(); });
    // and the zones beside it with bubbles on screen (T-276), once nothing waits on them
    effect(() => { nearbyZones.value; this.ensure(); });
    // T-351: a building tool picked in a game with no station readies the viewed zone for its solve
    effect(() => { tool.value; this.prefetchViewed(); });
  }

  /**
   * T-351: in a game with no station (a bare start), a building tool picked opens the viewed zone,
   * open bare, in every worker too, so the first station's solve need not wait for it (about 2.4 s
   * for New York in five workers). No solve covers it for that (`solveNeed`); it joins the solves
   * once a station reaches it, as any zone. Not while a game load's card is up.
   */
  private prefetchViewed() {
    const t = tool.peek();
    if ((t !== "track" && t !== "station" && t !== "blueprint") || perfOff("bareStart")) return;
    const svc = this.latest(), id = activeCity.peek();
    if (this.hold || !svc || svc.stations.length || !viewedZone.peek() || this.status.startsWith("failed")) return;
    if (!this.fetchingBare.has(id) || this.fetching.has(id) || this.fetchingUndecided.has(id)) return;
    this.prefetched.add(id);
    this.fetchCity(id, "full");
  }
  /** T-351: zones opened in every worker ahead of a station (`prefetchViewed`) */
  private prefetched = new Set<CityId>();

  /** T-276: nearby zones open for their bubbles only after the startup load and outside a loaded
   * game's clock hold, so they never hold the clock; they show no loading bar and a solve never
   * waits for them. T-325: they open bare unless a station reaches them. */
  private nearbyOpen(): boolean {
    return startup.peek() === "ready" && !this.hold;
  }

  // --- Loading only what is needed (T-149) -------------------------------------------------------

  /** T-318: the viewed zone alone; empty over dead space, where nothing loads for the view. */
  private viewedSet(): Set<CityId> {
    return new Set(viewedZone.peek() ? [activeCity.peek()] : []);
  }

  /** T-318: a network is in hand and neither the view nor any station needs a pack: nothing to
   * load or solve. The clock's hold goes, and no card or status says "Loading data". */
  private idle(): boolean {
    const svc = this.latest();
    return !!svc && this.needed(svc)?.size === 0;
  }

  private settleIdle() {
    demandUpdating.value = false; // a waiting network stays: it is solved when the view or a station needs a pack
    this.job = null;
    clearInterval(this.progressTick);
    this.progressTick = 0;
    loadProgress.value = null;
    const held = !!this.hold || startup.peek() === "loading";
    this.hold = null;
    if (held) loadComplete();
  }

  /** The cities a solve of `svc` needs open: the viewed one, and every one a built station reaches.
   * Null while that is not known yet (the footprints are still loading). */
  private needed(svc: Service | null): Set<CityId> | null {
    const out = this.viewedSet();
    if (!svc) return out;
    const reached = this.reachedSet(svc);
    if (!reached) return null;
    for (const id of reached) out.add(id);
    return out;
  }

  /** The cities `svc`'s stations reach; null while that is not known (no network yet, or the
   * footprints of zones whose boxes the stations come near still loading; T-404). A zone whose
   * footprint failed counts as reached. */
  private reachedSet(svc: Service | null): Set<CityId> | null {
    if (!svc) return null;
    if (svc.stations.length === 0) return new Set();
    let reached = this.reach.get(svc);
    if (reached) return reached;
    const { ll, near } = this.nearOf(svc);
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
    return reached;
  }

  private nearOf(svc: Service): { ll: Float64Array; near: CityId[]; markets: CityId[] } {
    let o = this.nearZones.get(svc);
    if (!o) {
      const f = makeFrame(svc.frame), ll = new Float64Array(2 * svc.stations.length);
      const markets = new Set<CityId>();
      svc.stations.forEach((s, k) => {
        const [lon, lat] = f.toLonLat(s.x, s.y);
        ll[2 * k] = lon; ll[2 * k + 1] = lat;
        for (const id of marketZonesNear(lon, lat)) markets.add(id);
      });
      this.nearZones.set(svc, (o = { ll, near: zonesNearStations(ll, REACH_M), markets: [...markets] }));
    }
    return o;
  }

  /** T-325: the cities a solve of `svc` must cover: every one its stations reach, and every one
   * opened in all workers (the viewed city at startup, before the reach was known). Null while
   * the reach is not known. */
  private solveNeed(svc: Service | null): Set<CityId> | null {
    const reached = this.reachedSet(svc);
    if (!reached) return null;
    const out = new Set(reached);
    // (T-351: not a zone opened ahead of a station that does not reach it yet)
    for (const id of this.fetching) if (!isMarket(id) && !this.prefetched.has(id as CityId)) out.add(id as CityId);
    return out;
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

  /** T-396, T-377: a zone's market pack is opened once the zone is viewed, beside the view or
   * reached, or a station is within its riders' reach (`marketZonesNear`); it closes as a city pack
   * does (`sweep`). */
  private marketWanted: CityId[] = [];
  private wantsMarket(svc: Service | null, need: Set<CityId>): CityId[] {
    const out = new Set<CityId>([...need, ...nearbyZones.peek(), ...(svc ? this.nearOf(svc).markets : [])]);
    this.marketWanted = CITIES.map((c) => c.id).filter((id) => out.has(id));
    return this.marketWanted;
  }

  /** Fetch whatever the newest service and the view need, the viewed city first; then solve if all is open. */
  private ensure() {
    if (this.status.startsWith("failed")) return;
    const known = this.needed(this.latest());
    const need = known ?? this.viewedSet();
    demandIdle.value = known?.size === 0; // T-318
    // over dead space with no station in reach there is nothing to open, the market included
    if (need.size) for (const id of this.wantsMarket(this.latest(), need)) if (!this.fetching.has(marketItem(id))) this.fetchMarket(id);
    // T-325: a city no station reaches opens bare; while the reach is not known (startup), the
    // viewed city's pack is fetched and decided once downloaded (T-351)
    const reached = this.knownReach();
    for (const id of need) this.want(id, reached === null ? null : reached.has(id));
    if (this.nearbyOpen()) for (const id of nearbyZones.peek()) this.want(id, !!reached?.has(id));
    this.flush();
    this.progress();
    this.sweep();
  }

  // --- Closing what is no longer needed (T-301) ---------------------------------------------------

  /** What must stay open: the zones viewed, beside the view, reached or in the load in hand, and the
   * market while one of those is its zone or a station is in its reach. Null while the reach is not
   * known. */
  private keepSet(): Set<string> | null {
    const reached = this.knownReach();
    if (!reached) return null;
    const keep = new Set<string>([...this.viewedSet(), ...nearbyZones.peek(), ...reached]);
    if (this.job) for (const id of this.job.items) keep.add(id);
    const svc = this.latest();
    for (const id of [...keep]) if (!isMarket(id)) keep.add(marketItem(id as CityId));
    if (svc) for (const id of this.nearOf(svc).markets) keep.add(marketItem(id));
    return keep;
  }

  /** Close the packs no longer needed (CLOSE_MS, KEEP_SPARE), and bare copies the solves have made
   * redundant. Not before startup has finished nor in a loaded game's hold; `?perfOff=close` never. */
  private sweep() {
    clearTimeout(this.closeTimer);
    this.closeTimer = 0;
    if (perfOff("close") || this.status.startsWith("failed") || this.ready < this.workers.length || this.held || startup.peek() !== "ready" || this.hold) return;
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

  /** Close zones open bare in the aux worker; their figures stay shown. */
  private closeBare(ids: CityId[]) {
    this.workers[this.computeN].postMessage({ kind: "close", ids, bare: true } satisfies ToDemand);
    for (const id of ids) {
      this.bareKept.set(id, this.bareOpen.get(id)!);
      this.bareOpen.delete(id);
      this.fetchingBare.delete(id);
      const log = lastLoad(this.start.loads, (l) => l.item === id && !!l.bare);
      if (log) log.closedAt = performance.now();
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
      this.acks.delete(id);
      this.fetching.delete(id);
      this.prefetched.delete(id as CityId);
      this.cityVersion.delete(id as CityId);
      const log = lastLoad(this.start.loads, (l) => l.item === id && !l.bare);
      if (log) log.closedAt = performance.now();
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

  /** Fetch `id` to open in every worker (`full`), bare in the aux worker (false), or (null: the
   * reach is not known yet) either, decided once downloaded; unless already. */
  private want(id: CityId, full: boolean | null) {
    if (this.fetching.has(id) || this.fetchingUndecided.has(id)) return;
    if (full) this.fetchCity(id, "full");
    else if (full === null) this.fetchCity(id, this.fetchingBare.has(id) || perfOff("bareStart") ? "full" : "undecided");
    else if (!this.fetchingBare.has(id)) this.fetchCity(id, "bare");
  }

  /** Download one pack file, counting its bytes as they arrive. */
  private async download(file: string): Promise<Uint8Array> {
    const r = await fetch(new URL(`packs/${file}`, document.baseURI));
    if (!r.ok) throw new Error(`${r.status} ${r.url}`);
    const d: Download = { got: 0, total: Number(r.headers.get("Content-Length")) || 0, at: performance.now() };
    this.downloads.set(file, d);
    if (!r.body) {
      const b = new Uint8Array(await r.arrayBuffer());
      d.got = d.total = b.length;
      return b;
    }
    const reader = r.body.getReader(), chunks: Uint8Array[] = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      d.got += value.length;
      this.progressSoon();
    }
    // a compressed response's length is not its body's: the body is what was downloaded
    d.total = d.got;
    if (chunks.length === 1) return chunks[0];
    const out = new Uint8Array(d.got);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }

  /** Post an `open` to every worker (the last takes the original buffer, the others copies).
   * T-173: until the first worker is ready, the other solving workers' city packs wait (`held`). */
  private post(open: { pack?: { id: CityId; header: string; bin: ArrayBuffer }; intercity?: { id: CityId; bytes: Uint8Array } }) {
    this.workers.forEach((w, i) => {
      const last = i === this.workers.length - 1;
      const pack = open.pack && { ...open.pack, bin: last ? open.pack.bin : open.pack.bin.slice(0) };
      // T-180: a market pack as bytes, each solving worker its own copy (the last solving one the
      // original), decoded there: decoding megabytes and cloning the text six times held the main
      // thread up while the player could already pan. The aux worker never opens one: empty bytes.
      const m = open.intercity;
      const bytes = !m ? undefined : i === this.computeN ? new Uint8Array(0) : i === this.computeN - 1 ? m.bytes : m.bytes.slice();
      const msg: ToDemand = { kind: "open", packs: pack ? [pack] : [], ...(m ? { intercity: [{ id: m.id, pack: bytes! }] } : {}) };
      const transfer: Transferable[] = pack ? [pack.bin] : [];
      if (bytes?.byteLength) transfer.push(bytes.buffer as ArrayBuffer);
      const send = () => w.postMessage(msg, transfer);
      if (pack && i > 0 && i < this.computeN && this.held) this.held.push(send);
      else send();
    });
  }
  /** T-173: worker 0 opens a small synthetic everyday layer before it is ready, and the other
   * solving workers get no city pack until then. With five workers opening New York's everyday
   * layer at once from the browser's baseline WebAssembly code, the optimised code came seconds
   * late (7-8 s in every worker instead of 1.5-2 s); compiled code is shared by a page's workers,
   * so after one worker's warm-up they all open on optimised code. Null once released (or with
   * `?perfOff=stagger`, the old way). */
  private held: (() => void)[] | null = perfOff("stagger") ? null : [];

  /** `full`: to every worker; `bare` (T-325): to the aux worker alone; `undecided` (T-351): either,
   * by the reach known once it has downloaded. A pack opened bare and reached later is fetched
   * again for every worker (the page keeps no copy). */
  private fetchCity(id: CityId, mode: "full" | "bare" | "undecided") {
    if (mode === "full") this.fetching.add(id);
    else if (mode === "bare") this.fetchingBare.add(id);
    else this.fetchingUndecided.add(id);
    const log: LoadLog = { item: id, requestedAt: performance.now(), ...(mode === "bare" ? { bare: true } : {}) };
    this.start.loads.push(log);
    Promise.all([this.download(`${id}.json`), this.download(`${id}.bin`)]).then(([header, bin]) => {
      log.downloadedAt = performance.now();
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
      if (!bare) return this.post({ pack });
      const msg: ToDemand = { kind: "open", packs: [pack], bare: true };
      this.workers[this.computeN].postMessage(msg, [own]);
    }).catch((err) => this.fail(`city pack ${id}: ${String(err)}`));
  }

  /** T-377: a zone's market pack, to every solving worker. */
  private fetchMarket(id: CityId) {
    const item = marketItem(id);
    this.fetching.add(item);
    const log: LoadLog = { item, requestedAt: performance.now() };
    this.start.loads.push(log);
    this.download(`${id}.market.json`).then((bytes) => {
      log.downloadedAt = performance.now();
      this.bytes.set(item, bytes.byteLength);
      this.hashPack(item, [bytes]);
      this.post({ intercity: { id, bytes } });
    }).catch((err) => this.fail(`market pack ${id}: ${String(err)}`));
  }

  /** A worker opened `item`: once all have, waiting solves and queries may go. */
  private openedBy(item: string) {
    const n = (this.acks.get(item) ?? 0) + 1;
    this.acks.set(item, n);
    if (n < this.workers.length) return;
    this.opened.add(item);
    const log = lastLoad(this.start.loads, (l) => l.item === item && !l.bare);
    if (log && log.openedAt === undefined) log.openedAt = performance.now();
    this.settleWaiters();
  }

  /** T-325: the aux worker opened zones bare: their figures and city views. */
  private bareOpened(figures: Record<CityId, BareFigures>) {
    for (const [id, f] of Object.entries(figures)) {
      this.bareOpen.set(id, f);
      this.bareKept.delete(id);
      const log = lastLoad(this.start.loads, (l) => l.item === id && !!l.bare);
      if (log && log.openedAt === undefined) log.openedAt = performance.now();
    }
    this.addBareViews();
    this.settleWaiters();
  }

  /** Resolve `whenOpen` waits for items now open (bare ones need only the aux worker). */
  private settleWaiters() {
    const all = this.ready === this.workers.length;
    this.onOpen = this.onOpen.filter((w) => ((this.opened.has(w.item) && all) || this.bareOpen.has(w.item) ? (w.f(true), false) : true));
  }

  /** Send the waiting service when everything it needs is open; or solve again for a city opened
   * since the newest solve (the viewed city after a pan). */
  private flush() {
    if (this.status.startsWith("failed")) return;
    if (this.idle()) return this.settleIdle();
    // T-325: a solve waits for, and covers, the cities stations reach and those opened in every
    // worker; a zone opened bare is never waited for and starts no solve
    if (this.waiting) {
      const need = this.solveNeed(this.waiting.svc);
      if (need && this.allOpen(need)) {
        const w = this.waiting;
        this.waiting = null;
        this.dispatch(w.svc, w.arrivedAt);
      }
      return;
    }
    const s = this.cur;
    const need = s && this.solveNeed(s.svc);
    if (!s || !need || !this.allOpen(need)) return;
    // a city opened since the newest solve (also one that finished loading after the view moved
    // away again: it is open, so solve it rather than leave its load unfinished)
    if ([...need].some((id) => !s.cities.includes(id)) || [...this.opened].some((item) => isMarket(item) && !s.markets.includes(marketZone(item)))) this.dispatch(s.svc, performance.now());
  }

  /** Resolves when `item` is open in every worker, or bare in the aux worker (false if the pool
   * failed). */
  private whenOpen(item: string): Promise<boolean> {
    if (this.status.startsWith("failed")) return Promise.resolve(false);
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
      if (isMarket(id) || shown.has(id) || next.has(id) || this.unsettled?.has(id) || this.status.startsWith("failed")) continue;
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
    if (s && this.unsettled?.size && s.published < ROUNDS && !this.status.startsWith("failed")) {
      let landed = 0;
      for (let r = 0; r <= ROUNDS; r++) landed += r <= s.published ? PERIODS.length : s.byRound[r].filter(Boolean).length;
      const stops = s.svc.lines.reduce((n, l) => n + l.stops.length, 0);
      const f = Math.max(landed / (PERIODS.length * (ROUNDS + 1)), timeShare(s.sentAt, solveEstimate(stops, "all"), now));
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

  /** The loading progress: each step weighted by its expected time, T-243 (game/loading.ts). */
  private jobProgress() {
    if (this.status.startsWith("failed")) return;
    const known = this.needed(this.latest());
    if (known?.size === 0) {
      // T-318: nothing to load or solve (dead space); the load in hand, if any, finishes unseen
      if (this.job) { this.job = null; clearInterval(this.progressTick); this.progressTick = 0; }
      loadProgress.value = null;
      return;
    }
    const need = known ?? this.viewedSet();
    const viewed = activeCity.peek();
    // T-325: a city the solves must cover is loaded when open in every worker; another, when open bare
    const sn = this.solveNeed(this.latest());
    const loaded = (item: string) => (isMarket(item) || sn === null || sn.has(item as CityId) ? this.opened.has(item) : this.isOpen(item));
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
      if (!missing.length && !(atStartup && marketsMissing.length) && !heldSolve) {
        if (!atStartup) loadProgress.value = null;
        return;
      }
      this.job = { items: [], cities: [], startup: atStartup, best: 0, version: heldSolve && !missing.length ? s!.version : undefined };
    }
    const job = this.job;
    for (const id of missing) if (!job.items.includes(id)) { job.items.push(id); job.cities.push(id); }
    for (const item of marketsMissing) if (!job.items.includes(item)) job.items.push(item);
    if (atStartup && need.has(viewed) && !job.cities.includes(viewed)) job.cities.unshift(viewed);
    const needsSolve = this.jobNeedsSolve(job);
    // (T-402: or, when the load's zones are all in the cluster solved ahead, that cluster's round;
    // T-421: a game load's, once the zone in view has its final)
    const ahead = s?.ahead && job.cities.every((id) => s.ahead!.cities.includes(id)) ? s.clusterRes?.get(s.ahead.key)?.published ?? -1 : -1;
    const ofJob = !!s && job.version !== undefined && s.version === job.version;
    const solved = !needsSolve || (ofJob && (Math.max(s.published, ahead) >= doneRound || (!!this.hold && this.viewHasFinal(s))));
    if (solved && job.items.every(loaded)) {
      // remembered for the next load's estimate (T-243): a startup's solve, so it describes the
      // game a reload opens; T-347, T-421: and the time to the card going, which its bar measures
      // (not a bare start's, T-351: its solve covers no zone and says nothing of the next game's)
      if (job.startup && s && s.cities.length) {
        const stops = s.svc.lines.reduce((n, l) => n + l.stops.length, 0);
        const at = s.refinedAt ?? (s.ahead && s.clusterRes?.get(s.ahead.key)?.at);
        const finalMs = at !== undefined ? Math.round(at - s.sentAt) : undefined;
        // T-421: a solve released at the view's final is remembered at its own final (`publish`)
        if (s.refinedAt === undefined) this.timeRest = { version: s.version, stops, finalMs };
        else if (s.firstAt !== undefined) rememberSolve(Math.round(s.firstAt - s.sentAt), stops, finalMs, finalMs);
      }
      this.job = null;
      clearInterval(this.progressTick);
      this.progressTick = 0;
      loadProgress.value = null;
      return;
    }
    if (!this.progressTick) this.progressTick = setInterval(() => this.progress(), 100) as unknown as number;
    const now = performance.now();
    const within = (since: number | undefined, expected: number) => timeShare(since, expected, now);
    // Downloading: bytes; expected to take as long as its rate so far says.
    let got = 0, total = 0, dlStart = Infinity;
    // Opening: each item in every worker, expected from its size; worker acks are its real progress.
    let openMs = 0, openDone = 0, openStart = Infinity;
    for (const item of job.items) {
      let size = 0;
      for (const f of isMarket(item) ? [`${marketZone(item)}.market.json`] : [`${item}.json`, `${item}.bin`]) {
        const d = this.downloads.get(f);
        if (d) { got += Math.min(d.got, d.total || d.got); total += d.total || d.got; dlStart = Math.min(dlStart, d.at); size += d.total || d.got; }
      }
      const ms = (this.bytes.get(item) ?? (size || GUESS_BYTES)) / 1e6 * (isMarket(item) ? OPEN_MS_PER_MB.market : OPEN_MS_PER_MB.city);
      openMs += ms;
      openDone += loaded(item) ? ms : ms * (this.acks.get(item) ?? 0) / this.workers.length;
      const log = lastLoad(this.start.loads, (l) => l.item === item);
      if (log?.downloadedAt !== undefined) openStart = Math.min(openStart, log.downloadedAt);
    }
    const downloaded = job.items.every((i) => this.bytes.has(i) || loaded(i));
    const dl = downloaded ? 1 : total > 0 ? Math.min(0.99, got / total) : 0;
    const dlElapsed = dlStart < Infinity ? now - dlStart : 0;
    const dlMs = downloaded ? dlElapsed : got > 0 && total > got ? dlElapsed * total / got : 300;
    const open = openMs > 0 ? Math.max(openDone / openMs, job.items.every(loaded) ? 1 : within(openStart < Infinity ? openStart : undefined, openMs)) : 1;
    // Solving: period results landed, and time against the expected solve (none for a load of
    // zones opened bare, T-325).
    const svc = this.latest();
    // (T-351: a solve covering no zone, the market alone, is short)
    const solveMs = !needsSolve ? 0 : sn?.size === 0 ? SOLVE_MS.base : solveEstimate(svc ? svc.lines.reduce((n, l) => n + l.stops.length, 0) : null, !!this.hold);
    let solve = 0;
    if (s && job.version !== undefined && s.version === job.version) {
      let landed = 0;
      for (let r = 0; r <= doneRound; r++) landed += r <= s.published ? PERIODS.length : s.byRound[r].filter(Boolean).length;
      solve = Math.max(landed / (PERIODS.length * (doneRound + 1)), within(s.sentAt, solveMs));
    }
    const doneMs = dlMs * dl + openMs * open + solveMs * solve;
    const fraction = Math.min(0.99, doneMs / (dlMs + openMs + solveMs));
    // When an expectation grows (the network arrives bigger than guessed, a download slows), the
    // true fraction falls below what is shown: rather than stand still until it catches up, the
    // rest of the work is spread over the rest of the bar from where it stands.
    let shown = job.anchor ? job.anchor.shown + (1 - job.anchor.shown) * Math.max(0, fraction - job.anchor.frac) / Math.max(1e-6, 1 - job.anchor.frac) : fraction;
    if (shown < job.best - 1e-6) { job.anchor = { shown: job.best, frac: fraction }; shown = job.best; }
    job.best = Math.max(job.best, Math.min(0.99, shown));
    const step = dl < 1 ? "Downloading city data" : open < 1 ? "Opening city data" : "Calculating demand";
    loadProgress.value = { cities: job.cities.filter((id) => !isMarket(id)), fraction: job.best, step, startup: job.startup };
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
   * have downloaded. */
  private async keyFor(cities: CityId[], markets: CityId[]): Promise<CacheKey> {
    const ids = [...markets.map(marketItem), ...cities];
    const hashes = await Promise.all(ids.map((id) => this.packHashOf(id).p));
    return { format: CACHE_FORMAT, build: BUILD, workers: this.computeN, flags: new URLSearchParams(location.search).get("perfOff") ?? "", packs: Object.fromEntries(ids.map((id, i) => [id, hashes[i]])) };
  }

  /** A whole game's network arrived: show its cached final result if there is one for exactly this
   * network, these packs and this build, and nothing of its solve is shown yet. */
  private async tryCache(svc: Service) {
    const log: NonNullable<DemandClient["cacheLog"]> = (this.cacheLog = { at: performance.now() });
    try {
      const canon = canonical(svc);
      const hash = (log.hash = await sha256(canon.text));
      const stored = await readDemandCache<StoredDemand>(hash, saveSlotsState.peek().active);
      log.readMs = performance.now() - log.at;
      if (!stored) return void (log.miss = "none");
      // the zones it covered must be the ones this solve will cover
      await loadFootprints(this.nearOf(svc).near);
      const need = this.solveNeed(svc);
      const cities = Object.keys(stored.packs ?? {}).filter((k) => !isMarket(k));
      if (!need || need.size !== cities.length || cities.some((id) => !need.has(id))) return void (log.miss = "zones");
      const markets = this.wantsMarket(svc, need);
      const had = Object.keys(stored.packs ?? {}).filter(isMarket).map(marketZone);
      if (had.length !== markets.length || markets.some((id) => !had.includes(id))) return void (log.miss = "zones");
      const why = sameKey(stored, await this.keyFor(cities, markets));
      if (why) return void (log.miss = why);
      if (this.latest() !== svc || (this.cur?.svc === svc && this.cur.published >= 0) || this.status.startsWith("failed")) return void (log.miss = "late");
      this.applyCache(svc, canon, stored, hash, cities);
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
    if (this.cur?.svc === svc) this.cur.showEstimates = false;
    demandByDay.value = days;
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
    const [hash, key] = await Promise.all([sha256(canon.text), this.keyFor(s.cities, s.markets)]);
    // the bubbles, when they were asked for this result (weekday, every purpose, the viewed zone)
    const m = this.modesP, city = this.focusCity;
    const cm = m && m.version === s.version && m.day === 0 ? await m.p : null;
    // the result shown at load is the same one, unless it can now add its bubbles
    if (hash === this.cachedHash && (this.cachedModes || !cm)) return;
    const modes = cm && { city, data: concat([cm.homeRail, cm.homeWalk, cm.homeDrive, cm.workRail, cm.workWalk, cm.workDrive]) };
    const ms = performance.now() - t0;
    const ok = await writeDemandCache(slots.active, hash, { ...key, at: Date.now(), round: s.published, days: stored, ...(modes ? { modes } : {}) } satisfies StoredDemand);
    this.cachedHash = hash;
    (this.cacheLog ??= { at: t0 }).wrote = { at: performance.now(), ms, bytes: storedSize(stored) + (modes?.data.byteLength ?? 0), ok };
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
    if (s.published >= HOLD_ROUND) return true;
    const a = s.ahead, c = a && s.clusterRes?.get(a.key);
    return !!c && c.published >= ROUNDS && [...this.viewedSet()].every((id) => a.cities.includes(id));
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
        this.job = null;
        clearInterval(this.progressTick);
        this.progressTick = 0;
        loadProgress.value = null;
        this.zoneProgress();
      }
      loadComplete();
      // the nearby zones' packs may load now (T-276)
      if (nearbyZones.peek().length) this.ensure();
      this.prefetchViewed(); // a building tool picked while the card was up (T-351)
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
    this.focusCity = c;
    this.focusEpoch++;
    this.cellsP = null;
    this.modesP = null;
    this.purposeP.clear();
  }

  private fail(why: string) {
    this.status = "failed: " + why;
    demandUpdating.value = false;

    this.onOpen.splice(0).forEach(w => w.f(false));
    for (const q of this.pendingQueries.values()) q.done([]);
    this.pendingQueries.clear();
    this.job = null;
    clearInterval(this.progressTick);
    this.progressTick = 0;
    this.hold = null;
    zoneLoads.value = new Map();
    console.error("demand:", why);
    loadFailed(why);
  }

  private receive(i: number, msg: FromDemand) {
    if (msg.kind === "failed") return this.fail(`worker ${i}: ${msg.error}`);
    if (msg.kind === "cells" || msg.kind === "subSums" || msg.kind === "cellModes" || msg.kind === "station" || msg.kind === "flowRail" || msg.kind === "flows" || msg.kind === "purposes" || msg.kind === "icSel" || msg.kind === "icStation" || msg.kind === "purposeModes" || msg.kind === "transfers" || msg.kind === "trip" || msg.kind === "bare" || msg.kind === "bareFlows" || msg.kind === "bareSquares") {
      const w = this.pendingQueries.get(msg.id);
      if (!w) return;
      const t = msg as { qStart?: number; qMs?: number };
      this.qlog.find((q) => q.id === msg.id)?.replies.push({ w: i, start: t.qStart ?? 0, ms: t.qMs ?? 0, at: performance.timeOrigin + performance.now() });
      w.replies.push(msg);
      if (--w.left === 0) {
        this.pendingQueries.delete(msg.id);
        w.done(w.replies);
      }
      return;
    }
    if (msg.kind === "sums") {
      const list = this.pushed.get(`${msg.version}:${msg.round}:${msg.city}`);
      if (list) list.push({ mask: msg.mask, sums: msg.sums, everyday: msg.everyday });
      else if (this.cur && msg.version === this.cur.version) this.pushed.set(`${msg.version}:${msg.round}:${msg.city}`, [{ mask: msg.mask, sums: msg.sums, everyday: msg.everyday }]);
      return;
    }
    if (msg.kind === "ready") {
      this.start.workers[i] = { wasmMs: msg.wasmMs, openMs: msg.openMs, warmMs: msg.warmMs, memBytes: msg.memBytes };
      // T-173: worker 0 has warmed the everyday code up; the held city packs may go
      if (i === 0 && this.held) { const held = this.held; this.held = null; held.forEach((send) => send()); }
      if (!periodsMatch(msg.periods, PERIOD_STRIDE))
        console.error("demand: game/periods.ts disagrees with sim/src/track/params.rs PERIOD_HOURS / PERIOD_LEVEL", msg.periods);
      if (!this.periodTable.length && i < this.computeN) this.periodTable = msg.periods;
      if (!this.dayTable.length && i < this.computeN) this.dayTable = msg.days;
      for (const id of Object.keys(msg.cityInfo)) this.openedBy(id);
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
      return;
    }
    if (msg.kind === "opened") {
      const w = this.start.workers[i];
      if (w) { w.openMs += msg.openMs; w.warmMs += msg.warmMs; w.memBytes = msg.memBytes; }
      if (msg.bare) {
        this.bareOpened(msg.bare);
        this.progress();
        this.checkHold();
        return;
      }
      for (const id of Object.keys(msg.cityInfo)) this.openedBy(id);
      for (const id of msg.markets) this.openedBy(marketItem(id));
      this.flush();
      this.progress();
      return;
    }
    const s = this.cur;
    if (!s || msg.version !== s.version) return; // superseded
    if (msg.kind === "done") {
      s.memBytes[i] = msg.memBytes;
      if (msg.clusters) s.clusters = msg.clusters; // ?debug=1's log: clusters kept and solved
      if (msg.periods) (s.periods ??= [])[i] = msg.periods;
      return;
    }
    if (msg.kind === "plan") {
      if (i === 0) {
        demandSolving.value = new Set(msg.solving.flat());
        if (msg.ahead) s.ahead = { cities: msg.ahead, key: msg.ahead.join(",") };
      }
      return;
    }
    if (msg.cluster) {
      // T-392: one cluster solved ahead of the rest: a round of it is shown once every period of it
      // is in. Its rounds land in order: with the barrier (T-402) no worker starts one before every
      // worker has the one before; after an edit only the last is sent.
      const key = msg.cluster.cities.join(",");
      const round = msg.summary[0];
      const m = (s.clusterRes ??= new Map());
      let e = m.get(key);
      if (!e) m.set(key, (e = { byRound: Array.from({ length: ROUNDS + 1 }, () => PERIODS.map(() => null)), published: -1 }));
      const res = e.byRound[round];
      if (!res) return;
      res[msg.period] = msg;
      if (!res.every(Boolean) || round <= e.published) return;
      e.published = round;
      if (round > 0) e.byRound[round - 1] = [];
      if (round < ROUNDS && s.ahead?.key === key) for (const w of this.workers.slice(0, this.computeN)) w.postMessage({ kind: "go", version: s.version, round: round + 1, group: 0 } satisfies ToDemand);
      if (round >= ROUNDS) e.at = performance.now();
      this.publishCluster(s, res as Result[], msg.cluster, round);
      this.progress();
      this.checkHold(); // T-421
      return;
    }
    const results = s.byRound[msg.summary[0]];
    if (!results) return;
    results[msg.period] = msg;
    (s.periodMs[msg.period] ??= [])[msg.summary[0]] = msg.ms;
    if (msg.setupMs) s.setupMs[i] = msg.setupMs;
    s.memBytes[i] = msg.memBytes;
    // Never publish a mixture of population/period rounds: fast workers may be ahead.
    for (let round = s.published + 1; round <= ROUNDS; round++) {
      if (!s.byRound[round].every(Boolean)) break;
      s.results = s.byRound[round];
      s.published = round;
      // T-327: every worker has the round, so the next may start
      if (round < ROUNDS) for (const w of this.workers.slice(0, this.computeN)) w.postMessage({ kind: "go", version: s.version, round: round + 1, group: s.ahead ? 1 : 0 } satisfies ToDemand);
      this.publish(s, round);
      if (round > 0) s.byRound[round - 1] = []; // release previous typed-array snapshots
    }
    this.progress();
    this.checkHold();
  }

  /** A network from the clock worker (T-025), or the test network. */
  submitNetwork(snap: NetworkSnapshot) {
    const svc = serviceFromSnapshot(snap);
    const key = JSON.stringify(svc);
    // the first network after a whole-game load is the one the clock waits for (T-149)
    if (takeGameLoadHold()) this.hold = {};
    if (key === this.lastKey) {
      // nothing demand reads changed: the solve in hand (or waiting) is the loaded game's
      if (this.hold && !this.waiting) this.hold.version = this.cur?.version;
      this.checkHold();
      return;
    }
    this.lastKey = key;
    demandUpdating.value = true;
    this.send(svc, performance.now());
    // T-328: a whole game's network: its last final result may be kept beside the save
    if (this.hold && this.cacheOn) void this.tryCache(svc);
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
    const { net, layout } = packService(svc);
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
    this.cur = { version, svc, layout, results: PERIODS.map(() => null), byRound: Array.from({length: ROUNDS + 1}, () => PERIODS.map(() => null)), published: -1, arrivedAt, sentAt: performance.now(), periodMs: [], setupMs: [], memBytes: [], cities, markets: [...this.opened].filter(isMarket).map(marketZone), showEstimates };
    this.log.push(this.cur);
    if (this.log.length > 50) this.log.shift();
    demandUpdating.value = true;
    demandStep.value = { done: 0, of: ROUNDS + 1 };
    // T-327: rounds in step only where clicks are answered by round (`target`): it costs the final
    // about 0.7 s on the real save
    demandSolving.value = null;
    const msg: ToDemand = { kind: "solve", version, net, rounds: ROUNDS, barrier: (showEstimates || !!this.hold) && !perfOff("ahead"), first: activeCity.peek(), ...(perfOff("clusters") ? { whole: true } : {}), ...(perfOff("split") ? { oneMarket: true } : {}), ...(perfOff("periods") ? { fresh: true } : {}) };
    for (const w of this.workers) w.postMessage(msg);
    this.progress(); // T-243: a held game's solve shows its progress even with nothing to load
  }

  // --- Demand views (T-078; types and meaning in game/demand.ts) ---------------------------------

  /** Send `make(id)` to the given workers (all by default) and collect their replies. */
  private ask(make: (id: number) => Query, which: number[] = this.workers.slice(0, this.computeN).map((_, i) => i), transfer: Transferable[] = [], zone?: CityId, t?: Target): Promise<Reply[]> {
    this.syncFocus();
    const id = this.nextQuery++;
    // `zone`: a nearby zone's question (T-276), about that zone whatever the focus does meanwhile
    const city = zone ?? activeCity.peek(), epoch = this.focusEpoch;
    return new Promise((done) => {
      this.pendingQueries.set(id, { left: which.length, replies: [], done: replies => done(zone !== undefined || epoch === this.focusEpoch ? replies : []) });
      const q = make(id);
      this.qlog.push({ id, kind: q.kind, sent: performance.timeOrigin + performance.now(), replies: [] });
      if (this.qlog.length > 200) this.qlog.shift();
      // T-260: the kind of day shown; each answer weighs its periods by it
      const round = t?.round ?? demandView.peek()?.round ?? -1;
      for (const i of which) this.workers[i].postMessage({...q, city, round, day: shownDay.peek(), ...(t?.wait ? { wait: true } : {})}, transfer);
    });
  }

  /**
   * T-327: where a view's question goes. While the running solve's estimates are shown (startup, a
   * zone just opened), or a cached result of the same network is, the workers are asked about the
   * newest round any of them is in, and each answers on reaching it: one round ahead of the shown
   * figures at most. Otherwise the shown result's version and round, as a worker holds them only
   * when idle. Not after an edit: the workers then hold a network the map does not show yet.
   */
  private target(): Target | null {
    const dv = demandView.peek();
    const s = this.cur;
    // T-402: the zone in view's cluster solved ahead counts its own rounds (the workers tell which
    // rounds a question means by its zone)
    const ahead = s?.ahead?.cities.includes(activeCity.peek()) ? s.clusterRes?.get(s.ahead.key)?.published ?? -1 : null;
    const published = ahead ?? s?.published ?? -1;
    // nothing shown yet (startup, no cache): asked of round 0, whose answer is in hand when it shows
    if (!dv) return s && published < 0 && !perfOff("ahead") ? { version: s.version, round: 0, view: s.version, wait: true, solve: s, shown: -1 } : null;
    const cv = this.cityVersion.get(activeCity.peek());
    if (s && cv === s.version) return { version: cv, round: ROUNDS, view: dv.version, wait: false, solve: s };
    const cached = dv.version === CACHE_VERSION && this.log[0]?.version === CACHE_VERSION && this.log[0].svc === s?.svc;
    if (s && published < ROUNDS && (dv.version === s.version || cached) && !perfOff("ahead")) {
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
    const load = (id: CityId) => this.whenOpen(id).then(async (ok): Promise<CityCells | null> => {
      if (!ok) return null;
      const [r] = await this.ask((q) => ({ kind: "cells", id: q, frame: frameOf() }), [this.computeN], [], other);
      if (r?.kind !== "cells" || r.xy.length === 0) return null;
      return { n: r.xy.length / 2, xy: r.xy, h3: r.h3, zone: r.zone, zoneM: r.zoneXY[0], zoneXY: r.zoneXY.subarray(1) };
    });
    if (other !== undefined) {
      // a nearby zone's (T-276): kept per zone and world frame; callers share the arrays and only read them
      const key = `${other}:${worldFrameEpoch.peek()}`;
      let p = this.otherCells.get(key);
      if (!p) {
        this.otherCells.set(key, (p = load(other)));
        p.then((v) => { if (!v && this.otherCells.get(key) === p) this.otherCells.delete(key); });
      }
      return p;
    }
    // waits for the viewed city to open (T-149: it may be loading after a pan)
    return (this.cellsP ??= load(activeCity.peek()));
  }
  /** Nearby zones' cells (T-276), `zone:frame epoch`. */
  private otherCells = new Map<string, Promise<CityCells | null>>();

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
    const at: Target = src !== undefined ? { version: src, round: ROUNDS, view: version, wait: false } : { version, round, view: version, wait: false };
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
      let modes = new Float32Array(6 * cells.n);
      const bare = this.isBare(city);
      for (const kind of kinds) {
        if (bare) {
          // T-325: the aux worker's no-rail pass, one part per solving worker's periods
          const [r] = await this.ask((id) => ({ kind: "bare", id, version, purpose: kind }), [this.computeN], [], other);
          if (r?.kind !== "bare" || !r.commute) return null;
          for (const p of bareParts(r)) if (p.length === modes.length) for (let k = 0; k < modes.length; k++) modes[k] += p[k];
          continue;
        }
        let mask = 0;
        for (const r of byMask(await this.ask((id) => ({ kind: "purposeModes", id, version: at.version, purpose: kind }), undefined, [], other, at))) {
          if (r.kind !== "purposeModes" || !r.modes) return null;
          mask |= r.mask;
          if (r.modes.length === modes.length) for (let k = 0; k < modes.length; k++) modes[k] += r.modes[k];
        }
        if (mask !== ALL_PERIODS) return null;
      }
      if ((other === undefined && activeCity.peek() !== city) || demandView.peek()?.version !== version || demandView.peek()?.round !== round || shownDay.peek() !== day) return null;
      const n = cells.n;
      const b = (k: number) => modes.subarray(k * n, (k + 1) * n);
      return { version, day, homeRail: b(0), homeWalk: b(1), homeDrive: b(2), workRail: b(3), workWalk: b(4), workDrive: b(5) };
    })();
    if (other !== undefined) return p;
    // one solve's purposes at a time; a failed answer is not kept
    for (const k of this.purposeP.keys()) if (!k.startsWith(`${version}:${round}:`)) this.purposeP.delete(k);
    this.purposeP.set(key, p);
    p.then((v) => { if (!v && this.purposeP.get(key) === p) this.purposeP.delete(key); });
    return p;
  }

  cellModes(everydayToo = true, zone?: CityId): Promise<CellModes | null> {
    this.syncFocus();
    // `zone`: a nearby zone's legs (T-276), not cached here
    const other = zone !== undefined && zone !== activeCity.peek() ? zone : undefined;
    const version = demandView.peek()?.version;
    if (version === undefined) return Promise.resolve(null);
    // cached per solve and crowding round (T-084: the dot map asks again for the refined result)
    const round = demandView.peek()?.round ?? 0;
    const day = shownDay.peek();
    // T-392: a zone shown from a cluster solved ahead of its solve's final is asked about there
    const src = this.cityVersion.get(other ?? activeCity.peek());
    const at: Target = src !== undefined ? { version: src, round: ROUNDS, view: version, wait: false } : { version, round, view: version, wait: false };
    if (everydayToo && other === undefined && this.modesP?.version === version && this.modesP.round === round && this.modesP.day === day && this.modesP.src === src) return this.modesP.p;
    // T-328: the cached result's bubbles, when it kept this zone's for this day
    const kept = this.cachedModes;
    if (version === CACHE_VERSION && everydayToo && other === undefined && day === 0 && kept?.city === activeCity.peek()) {
      const n = kept.data.length / 6, b = (k: number) => kept.data.slice(k * n, (k + 1) * n);
      return Promise.resolve({ version, day, homeRail: b(0), homeWalk: b(1), homeDrive: b(2), workRail: b(3), workWalk: b(4), workDrive: b(5) });
    }
    const p = (async (): Promise<CellModes | null> => {
      const city = other ?? activeCity.peek();
      if (this.isBare(city)) return this.bareModes(city, everydayToo, version, round, day, other);
      let mask = 0;
      let sums: Float32Array | null = null;
      let extra: Float32Array | null = null;
      // every worker sent its sums with the round (no question to a worker busy with the next
      // round); asked for only when one is missing. A weekend's are always asked for (T-260).
      const sent = day === 0 ? this.pushed.get(`${at.version}:${at.round}:${city}`) : undefined;
      const parts: { mask: number; sums: Float32Array | null; everyday: Float32Array | null }[] = sent && sent.length === this.computeN ? sent : (await this.ask((id) => ({ kind: "subSums", id, version: at.version }), undefined, [], other, at)).map((r) => (r.kind === "subSums" ? r : { mask: 0, sums: null, everyday: null }));
      // in worker (period) order, not arrival order, so the same result gives the same bubbles to
      // the bit (T-328: the cached bubbles are compared with the fresh ones)
      for (const r of [...parts].sort((a, b) => a.mask - b.mask)) {
        if (!r.sums) return null;
        mask |= r.mask;
        if (!sums) sums = r.sums.slice();
        else for (let k = 0; k < sums.length; k++) sums[k] += r.sums[k];
        if (r.everyday) {
          if (!extra) extra = r.everyday.slice();
          else for (let k = 0; k < extra.length; k++) extra[k] += r.everyday[k];
        }
      }
      if (!sums || mask !== ALL_PERIODS) return null;
      const [r] = await this.ask((id) => ({ kind: "cellModes", id, version: at.version, sums: sums! }), [this.computeN], [sums.buffer], other, at);
      if (r?.kind !== "cellModes" || !r.modes) return null;
      if ((other === undefined && activeCity.peek() !== city) || demandView.peek()?.version !== version || demandView.peek()?.round !== round || shownDay.peek() !== day) return null;
      // Bubbles now count trip legs at each home/activity pair, both directions.
      // Old commuter queries count one person; convert before adding everyday travel.
      // (work alone for the purpose filter, T-158: `everydayToo` false)
      for (let k = 0; k < r.modes.length; k++) r.modes[k] = 2 * r.modes[k] + (everydayToo ? extra?.[k] ?? 0 : 0);
      const n = r.modes.length / 6;
      const b = (k: number) => r.modes!.subarray(k * n, (k + 1) * n);
      return { version, day, homeRail: b(0), homeWalk: b(1), homeDrive: b(2), workRail: b(3), workWalk: b(4), workDrive: b(5) };
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
      else for (let k = 0; k < extra.length; k++) extra[k] += p[k];
    }
    if ((other === undefined && activeCity.peek() !== city) || demandView.peek()?.version !== version || demandView.peek()?.round !== round || shownDay.peek() !== day) return null;
    const m = r.commute;
    for (let k = 0; k < m.length; k++) m[k] = 2 * m[k] + (everydayToo ? extra?.[k] ?? 0 : 0);
    const n = m.length / 6;
    const b = (k: number) => m.subarray(k * n, (k + 1) * n);
    return { version, day, homeRail: b(0), homeWalk: b(1), homeDrive: b(2), workRail: b(3), workWalk: b(4), workDrive: b(5) };
  }

  /** A station's riders, or a station complex's (`members`, T-146): every member is asked, and
   * members the demand model merged into one station (they answer identically) count once. */
  stationRiders(stationId: string, top = 20, members: string[] = [stationId]): Promise<StationRiders | null> {
    return this.viaRound(`station|${members.join(",")}|${top}`, (t) => this.stationRidersAt(t, stationId, top, members));
  }
  private async stationRidersAt(t: Target, stationId: string, top: number, members: string[]): Promise<StationRiders | null> {
    const solve = t.solve, version = t.view;
    if (!solve) return null;
    const indices = members.map((m) => solve.svc.stations.findIndex((s) => s.id === m)).filter((i) => i >= 0);
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
    const at: { station: string; line: string }[] = [];
    for (const { id, o, n, stops } of solve.layout)
      for (let k = 0; k < n; k++) {
        // stop k is route node o + k on run 0 and o + 2n - 1 - k on run 1
        at[o + k] = at[o + 2 * n - 1 - k] = { station: stops[k], line: id };
        if (inside.has(stops[k])) nodes.push(o + k, o + 2 * n - 1 - k);
      }
    if (nodes.length === 0) return null;
    const ids = Uint32Array.from(nodes);
    const sum = new Map<string, TransferPair>();
    let mask = 0;
    for (const r of byMask(await this.ask((id) => ({ kind: "transfers", id, version: t.version, nodes: ids }), undefined, [], undefined, t))) {
      if (r.kind !== "transfers" || !r.pairs) return null;
      mask |= r.mask;
      const v = r.pairs;
      for (let k = 0; k + 2 < v.length; k += 3) {
        const a = at[v[k]], b = at[v[k + 1]];
        if (!a || !b) continue;
        const key = `${a.station}|${a.line}|${b.station}|${b.line}`;
        const p = sum.get(key);
        if (p) p.riders += v[k + 2];
        else sum.set(key, { fromStation: a.station, fromLine: a.line, toStation: b.station, toLine: b.line, riders: v[k + 2] });
      }
    }
    if (mask !== ALL_PERIODS || !this.still(t)) return null;
    return { version, round, scope: members.join(","), pairs: [...sum.values()].sort((x, y) => y.riders - x.riders) };
  }

  /** The trip explorer (T-195; game/trip.ts): one trip from `from` to `to` (world metres) as a
   * commute, every period from the worker holding it. Null while there is no solve, or the
   * workers have moved on to a newer network or round. */
  trip(from: [number, number], to: [number, number]): Promise<Trip | null> {
    return this.viaRound(`trip|${from}|${to}`, (t) => this.tripAt(t, from, to));
  }
  private async tripAt(t: Target, from: [number, number], to: [number, number]): Promise<Trip | null> {
    const solve = t.solve, version = t.view, round = t.round;
    if (!solve) return null;
    const at: RouteNodes = [];
    for (const { id, o, n, stops } of solve.layout)
      for (let k = 0; k < n; k++) at[o + k] = at[o + 2 * n - 1 - k] = { station: stops[k], line: id, k };
    const periods: Trip["periods"] = PERIODS.map(() => null);
    let mask = 0, outside = false;
    for (const r of await this.ask((id) => ({ kind: "trip", id, version: t.version, from, to }), undefined, [], undefined, t)) {
      if (r.kind !== "trip" || !r.answers) return null;
      mask |= r.mask;
      r.periods.forEach((q, i) => {
        const v = r.answers![i];
        if (v[0] === -1) outside = true;
        periods[q] = readTripPeriod(v, at);
      });
    }
    if (mask !== ALL_PERIODS || !this.still(t)) return null;
    return { version, round, key: tripKey(from, to), periods, outside };
  }

  /** Where the commuters of a set of cells go (T-098; game/demand.ts `Flows`): every worker's rail
   * part of its periods, added up, then one worker spreads all modes over the far end. */
  flows(end: "home" | "work", cells: Uint32Array): Promise<Flows | null> {
    if (cells.length === 0) return Promise.resolve(null);
    // a zone open bare has no rounds (T-325)
    if (this.isBare(activeCity.peek())) {
      const dv = demandView.peek();
      return dv ? this.flowsAt({ version: dv.version, round: dv.round, view: dv.version, wait: false }, end, cells) : Promise.resolve(null);
    }
    return this.viaRound(`flows|${end}|${cells.join(",")}`, (t) => this.flowsAt(t, end, cells));
  }
  private async flowsAt(t: Target, end: "home" | "work", cells: Uint32Array): Promise<Flows | null> {
    const city = activeCity.peek(), round = demandView.peek()?.round;
    const version = t.view;
    const e = end === "home" ? 0 : 1;
    // the commutes' far end, packed as `flows` answers, and each worker's `purposes` answer
    let p: Float32Array;
    let purposeParts: { mask: number; work: Float32Array | null; everyday: Float32Array | null; names: string[] }[];
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
        else for (let k = 0; k < sums.length; k++) sums[k] += r.sums[k];
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
    purposeParts.sort((a, b) => a.mask - b.mask);
    const n = p[6];
    const at = (k: number) => p.subarray(7 + k * n, 7 + (k + 1) * n);
    const result: Flows = {
      version,
      end,
      rail: 2*p[0],
      walk: 2*p[1],
      drive: 2*p[2],
      cutRail: 2*p[3],
      cutWalk: 2*p[4],
      cutDrive: 2*p[5],
      cells: Uint32Array.from(at(0)),
      farRail: Float32Array.from(at(1), v => 2*v),
      farWalk: Float32Array.from(at(2), v => 2*v),
      farDrive: Float32Array.from(at(3), v => 2*v),
      purposes: [],
    };
    // each purpose on its own for the map's filter (T-158): work is the commute answer as it is
    const { rail, walk, drive, cutRail, cutWalk, cutDrive, cells: farCells, farRail, farWalk, farDrive } = result;
    const byPurpose = new Map<PurposeKey, FarEnd>([["work", { rail, walk, drive, cutRail, cutWalk, cutDrive, cells: farCells, farRail, farWalk, farDrive }]]);
    result.byPurpose = byPurpose;
    let work: Float32Array | null = null;
    let purposeNames: string[] | null = null;
    // T-285: each worker's everyday answer is packed (per purpose kind 30 timing values, m, m far
    // cells ascending, their rail, walk and drive legs), only the far cells the selection reaches
    const everyday: Float32Array[] = [];
    let mask = 0;
    for (const reply of purposeParts) {
      if (!reply.work) return null;
      if (purposeNames && JSON.stringify(purposeNames) !== JSON.stringify(reply.names)) return null;
      purposeNames = reply.names;
      mask |= reply.mask;
      if (!work) work = reply.work.slice();
      else for (let k=0;k<work.length;k++) work[k] += reply.work[k];
      if (reply.everyday) everyday.push(reply.everyday);
    }
    if (mask !== ALL_PERIODS || !work) return null;
    const purpose = (name: string, values: Float32Array) => ({name,
      outward: [0,1,2].map(m => Array.from(values.subarray(m*5,m*5+5))),
      returning: [0,1,2].map(m => Array.from(values.subarray(15+m*5,20+m*5))),
    });
    result.purposes.push(purpose("Work", work));
    if (everyday.length) {
      const kinds = purposeNames!.length;
      if (!kinds) return null;
      // per shown purpose (school counts university too, T-192): far-end legs by mode, and totals
      const merged = mergeEveryday(everyday, kinds, (kind) => (EVERYDAY_KEYS[kind] ? shownKey(EVERYDAY_KEYS[kind]) : null), result, [result.rail, result.walk, result.drive]);
      if (!merged) return null;
      merged.timing.forEach((t, kind) => result.purposes.push(purpose(purposeNames![kind], t)));
      for (const [key, { far, total }] of merged.shown)
        byPurpose.set(key as PurposeKey, { rail: total[0], walk: total[1], drive: total[2], cutRail: 0, cutWalk: 0, cutDrive: 0, ...far });
      Object.assign(result, merged.far);
      [result.rail, result.walk, result.drive] = merged.legs;
    }
    if (activeCity.peek() !== city || !this.still(t) || (!t.wait && demandView.peek()?.round !== round)) return null;
    return result;
  }

  /** Add up every worker's answer to a longer-trip query (each holds some of the periods); null
   * unless all five periods answered at the version asked about. `matched` copies the first
   * answer's first number (the same on every worker). */
  private async sumIntercity(kind: "icSel" | "icStation", t: Target, make: (id: number) => Query): Promise<Float64Array | null> {
    const parts = await this.ask(make, undefined, [], undefined, t);
    let mask = 0;
    let sums: Float64Array | null = null;
    // in worker (period) order, not arrival order: the same sums every time (T-146 compares them)
    for (const r of byMask(parts)) {
      if (r.kind !== kind || !r.packed) return null;
      mask |= r.mask;
      if (!sums) sums = r.packed.slice();
      else addIntercity(kind, sums, r.packed);
    }
    if (!this.still(t) || mask !== ALL_PERIODS) return null;
    return sums;
  }

  /** The longer trips of a bubble selection (game/demand.ts `IntercitySelection`, T-118). */
  intercitySelection(end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null> {
    if (cells.length === 0) return Promise.resolve(null);
    return this.viaRound(`icSel|${end}|${cells.join(",")}`, (t) => this.intercitySelectionAt(t, end, cells));
  }
  private async intercitySelectionAt(t: Target, end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null> {
    const version = t.view;
    const e = end === "home" ? 0 : 1;
    // T-336: a zone open bare: the solving workers hold the market but not the zone, so its squares
    // come from the aux worker with the question
    let sq: { xy: Float32Array; share: Float32Array } | undefined;
    if (this.isBare(activeCity.peek())) {
      const [r] = await this.ask((id) => ({ kind: "bareSquares", id, version, end: e, cells }), [this.computeN]);
      if (r?.kind !== "bareSquares" || !r.xy || !r.share) return null;
      sq = { xy: r.xy, share: r.share };
    }
    const v = await this.sumIntercity("icSel", t, (id) => ({ kind: "icSel", id, version: t.version, end: e, cells, ...(sq ? { sq } : {}) }));
    if (!v) return null;
    // per place, 8 numbers after the matched share: starts (all, rail, x, y) then ends; then per
    // far zone (all, rail) of starts and of ends, then zone x, y, market (not summed)
    const P = INTERCITY_PLACES.length;
    const head = 1 + 8 * P, nz = intercityZones("icSel", v.length), xy = intercityZoneXY("icSel", v.length);
    // T-276: the commutes between zones after the zones' x, y: per place, then per zone, the legs of
    // commuters living in the selection (by where they work) and of those working there (by where
    // they live), all and rail. Starts take the first, ends the second.
    const tail = xy + 3 * nz;
    const own = CITIES.find((c) => c.id === activeCity.peek())?.place ?? -1;
    const flows = (dir: number): PlaceFlows => {
      const places: PlaceTrips[] = [];
      let all = 0, rail = 0, air = 0, car = 0;
      for (let p = 0; p < P; p++) {
        const o = 1 + 8 * p + 4 * dir;
        const ca = v[tail + 4 * p + 2 * dir] ?? 0, cr = v[tail + 4 * p + 2 * dir + 1] ?? 0;
        const a = v[o] + ca, r = v[o + 1] + cr;
        // T-378: the longer trips not by rail by the corridor's shares; commuters drive
        const [pa, pc] = own >= 0 ? corridorSplit(this.corridorModes, corridorOf(own, p), v[o] - v[o + 1]) : [0, v[o] - v[o + 1]];
        const c = pc + ca - cr;
        all += a; rail += r; air += pa; car += c;
        if (a > 0) {
          const city = CITIES.find((c) => c.place === p);
          places.push({ place: p, all: a, rail: r, air: pa, car: c, ...(v[o] > 0 || !city ? marketXY(v[o + 2] / Math.max(v[o], 1e-9), v[o + 3] / Math.max(v[o], 1e-9)) : centreXY(city.center)) });
        }
      }
      const zones: ZoneTrips[] = [];
      for (let z = 0; z < nz; z++) {
        const c = tail + 4 * P + 4 * z + 2 * dir;
        const a = v[head + 4 * z + 2 * dir] + (v[c] ?? 0);
        if (a > 0) zones.push({ ...marketXY(v[xy + 3 * z], v[xy + 3 * z + 1]), place: v[xy + 3 * z + 2], all: a, rail: v[head + 4 * z + 2 * dir + 1] + (v[c + 1] ?? 0) });
      }
      return { all, rail, air, car, elsewhere: v[v.length - 2 + dir] ?? 0, places: places.sort((a, b) => b.all - a.all), zones: zones.sort((a, b) => b.all - a.all) };
    };
    return { version, starts: flows(0), ends: flows(1), matched: v[0] };
  }

  /** A station's longer-trip rail riders (game/demand.ts `IntercityStation`, T-118), or a station
   * complex's (`members`, T-146), each demand station once as in `stationRiders`. */
  intercityStation(stationId: string, members: string[] = [stationId]): Promise<IntercityStation | null> {
    return this.viaRound(`icStation|${members.join(",")}`, (t) => this.intercityStationAt(t, stationId, members));
  }
  private async intercityStationAt(t: Target, stationId: string, members: string[]): Promise<IntercityStation | null> {
    const solve = t.solve, version = t.view;
    if (!solve) return null;
    const indices = members.map((m) => solve.svc.stations.findIndex((s) => s.id === m)).filter((i) => i >= 0);
    if (indices.length === 0) return null;
    const answers = await Promise.all(indices.map((station) => this.sumIntercity("icStation", t, (id) => ({ kind: "icStation", id, version: t.version, station }))));
    let v: Float64Array | null = null;
    const seen = new Set<string>();
    for (const a of answers) {
      if (!a) continue;
      const sig = signature(a);
      if (seen.has(sig)) continue;
      seen.add(sig);
      if (!v) v = a.slice();
      else addIntercity("icStation", v, a);
    }
    if (!v) return null;
    const n = INTERCITY_PLACES.length;
    // rows: counts by place for starts then ends, then those times x and y (starts x, y, ends x, y);
    // then per far zone the riders starting, then ending; then zone x, y, market (not summed)
    const head = 6 * n, nz = (v.length - head) / 5, xy = intercityZoneXY("icStation", v.length);
    const flows = (dir: number): PlaceFlows => {
      const places: PlaceTrips[] = [];
      let rail = 0;
      for (let p = 0; p < n; p++) {
        const c = v[dir * n + p];
        rail += c;
        if (c > 0) places.push({ place: p, all: c, rail: c, air: 0, car: 0, ...marketXY(v[(2 + 2 * dir) * n + p] / c, v[(3 + 2 * dir) * n + p] / c) });
      }
      const zones: ZoneTrips[] = [];
      for (let z = 0; z < nz; z++) {
        const c = v[head + dir * nz + z];
        if (c > 0) zones.push({ ...marketXY(v[xy + 3 * z], v[xy + 3 * z + 1]), place: v[xy + 3 * z + 2], all: c, rail: c });
      }
      return { all: rail, rail, air: 0, car: 0, elsewhere: 0, places: places.sort((a, b) => b.all - a.all), zones: zones.sort((a, b) => b.all - a.all) };
    };
    return { version, stationId, scope: members.join(","), starts: flows(0), ends: flows(1) };
  }

  /** Every period has reached `round`: build the view the UI reads. */
  private publish(s: Solve, round: number) {
    if (round === 0) s.firstAt = performance.now();
    if (s === this.cur) demandStep.value = { done: round + 1, of: ROUNDS + 1 };
    // T-232: a round before the final one is shown only when nothing final would be held instead
    const final = round >= ROUNDS;
    const res = s.results as Result[];
    if (!final && !s.showEstimates) {
      // T-324: lines new or changed since the shown result have no figures; theirs come from the
      // first estimate at once, every other figure waits for the final round as above
      if (round === 0 && s === this.cur && !perfOff("early")) this.publishNewLines(s, res);
      return;
    }
    // T-260: the weekday is the solve itself; a weekend is worked out from it, purpose by purpose
    demandByDay.value = DEMAND_DAYS.map((day) => this.dayDemand(s, round, final, day, day === 0 ? res : res.map((r, q) => this.dayResult(r, q, day))));
    if (final) {
      s.refinedAt = performance.now();
      this.aheadP.clear();
      this.cityVersion.clear();
      if (s === this.cur) {
        demandUpdating.value = false;
        demandSolving.value = null;
        this.unsettled = null;
        void this.writeCache(s); // T-328
      }
      const t = this.timeRest;
      if (t?.version === s.version) {
        this.timeRest = undefined;
        if (s.firstAt !== undefined) rememberSolve(Math.round(s.firstAt - s.sentAt), t.stops, t.finalMs, Math.round(s.refinedAt - s.sentAt));
      }
    }
  }

  /**
   * T-324: the first estimate's figures for the lines the shown result has none for that fit (new
   * lines, and lines whose stops changed), added to the shown result. Everything else shown stays
   * as it is: other lines, the split, the bubbles, and stations a shown line already serves (a
   * station only the new lines serve gets the estimate too). The view keeps the shown solve's
   * version and round, so queries and the bubbles behave as before; the final round replaces all.
   */
  private publishNewLines(s: Solve, res: Result[]) {
    const days = demandByDay.peek();
    const base = this.log.find((x) => x.version === days[0].view?.version);
    if (!base || !days[0].view) return;
    const had = new Map(base.layout.map((l) => [l.id, l.stops.join(",")]));
    // a line given early figures by an earlier solve's estimate is still "new" against the base
    const fresh = s.layout.filter((l) => had.get(l.id) !== l.stops.join(","));
    if (!fresh.length) return;
    const part = { ...s, layout: fresh };
    s.earlyAt = performance.now();
    s.earlyLines = fresh.map((l) => l.id);
    demandByDay.value = days.map((d, k) => {
      const day = k as DemandDay;
      if (!d.view) return d;
      const est = this.dayDemand(part, 0, false, day, day === 0 ? res : res.map((r, q) => this.dayResult(r, q, day))).view!;
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
  }

  /**
   * T-392: a cluster's final figures, solved ahead of the rest of the solve, put in the shown result:
   * its lines and their stations, its zones' views, and the regional market's if it is in it. The
   * other figures stay as they are, the network-wide totals among them, until the solve's final;
   * the view keeps the shown version, and questions about the cluster's zones go to this solve
   * (`cityVersion`), whose workers hold it. T-402: `round` below the last comes from a cluster
   * solved ahead in a solve with the barrier.
   */
  private publishCluster(s: Solve, res: Result[], c: { cities: CityId[]; market: boolean; lines: Uint32Array }, round: number) {
    if (s !== this.cur) return;
    const final = round >= ROUNDS;
    const days = demandByDay.peek();
    const fresh = Array.from(c.lines, (l) => s.layout[l]).filter(Boolean);
    const part = { ...s, layout: fresh };
    const of = (day: DemandDay, final: boolean) => this.dayDemand(part, round, final, day, day === 0 ? res : res.map((r, q) => this.dayResult(r, q, day)));
    // T-402: with nothing shown yet, or only this solve's own figures of the cluster, a view of the
    // cluster alone, which the solve's own results replace (its totals are the cluster's, so never
    // the solve's final); its rounds before the last only where the solve's estimates are shown
    const alone = !days[0].view || days[0].view.version === s.version;
    if (!final && (!s.showEstimates || !alone)) return;
    if (alone) demandByDay.value = DEMAND_DAYS.map((day) => of(day, false));
    else {
      const ids = new Set(s.layout.map((l) => l.id));
      demandByDay.value = days.map((d, k) => {
        const day = k as DemandDay;
        if (!d.view) return d;
        const est = of(day, final);
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
    if (!final) return;
    for (const id of c.cities) this.cityVersion.set(id, s.version);
    const left = new Set(demandSolving.peek() ?? []);
    for (const id of c.cities) { left.delete(id); this.unsettled?.delete(id); }
    demandSolving.value = left;
  }

  /**
   * T-325: a bare zone's view of one kind of day, worked out from its figures exactly as
   * `dayResult` and `dayDemand` work out a solved city's (the figures are what a solve gives it).
   */
  private bareView(id: CityId, day: DemandDay, network: DemandView): DemandView {
    const b = (this.bareOpen.get(id) ?? this.bareKept.get(id))!;
    let trips = 0, rail = 0, walk = 0;
    const trainShareByPeriod: [string, number][] = PERIODS.map((period, q) => {
      let s = b.summary[q];
      if (day !== 0 && this.dayTable.length) {
        s = s.slice();
        const cp = b.parts[q];
        s[1] = s[2] = s[3] = 0;
        for (let i = 0; i * 3 < cp.length; i++) {
          const f = this.factor(day, b.keys[i], q);
          s[1] += cp[3 * i] * f; s[2] += cp[3 * i + 1] * f; s[3] += cp[3 * i + 2] * f;
        }
      }
      const [, t, ra, wa] = s;
      trips += t; rail += ra; walk += wa;
      return [period.name, t > 0 ? 100 * ra / t : 0];
    });
    const pct = (n: number) => trips > 0 ? 100 * n / trips : 0;
    return { ...network, tripsPerDay: trips, railTripsPerDay: rail, trainShareByPeriod, split: { train: pct(rail), walk: pct(walk), drive: Math.max(0, 100 - pct(rail) - pct(walk)) } };
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

  /** A kind of day's load factor for purpose `key` in period `q` (sim/src/demand/days.rs). */
  private factor(day: DemandDay, key: string, q: number): number {
    const p = PURPOSE_INDEX[key as PurposeKey] ?? 0;
    return this.dayTable[(day * PURPOSE_INDEX_COUNT + p) * PERIODS.length + q] ?? 1;
  }

  /**
   * One period's result as it is on another kind of day (T-260): each purpose's riders times its
   * load factor, the totals their sum, the mode split from each city's purposes the same way, the
   * market's trips and corridors by its own factor. The busiest-hour load of crush follows the
   * segment's riders. Route and mode choices are the weekday's (docs/demand.md 4.2).
   */
  private dayResult(r: Result, q: number, day: DemandDay): Result {
    const p = r.parts;
    if (!p || !this.dayTable.length) return r;
    const nr = r.seg.length;
    const seg = new Float32Array(nr), board = new Float32Array(nr), alight = new Float32Array(nr);
    const ps = new Float32Array(p.seg.length), pb = new Float32Array(p.board.length), pa = new Float32Array(p.alight.length);
    p.keys.forEach((key, i) => {
      const f = this.factor(day, key, q);
      for (let j = 0, o = i * nr; j < nr; j++, o++) {
        ps[o] = p.seg[o] * f; pb[o] = p.board[o] * f; pa[o] = p.alight[o] * f;
        seg[j] += ps[o]; board[j] += pb[o]; alight[j] += pa[o];
      }
    });
    const loadOfCrush = r.loadOfCrush.map((v, j) => (r.seg[j] > 0 ? (v * seg[j]) / r.seg[j] : 0));
    const citySummary: Record<CityId, number[]> = {};
    for (const [id, s0] of Object.entries(r.citySummary)) {
      const s = s0.slice(), cp = r.cityParts?.[id];
      if (cp) {
        s[1] = s[2] = s[3] = 0;
        for (let i = 0; i * 3 < cp.length; i++) {
          const f = this.factor(day, p.keys[i], q);
          s[1] += cp[3 * i] * f; s[2] += cp[3 * i + 1] * f; s[3] += cp[3 * i + 2] * f;
        }
      }
      citySummary[id] = s;
    }
    const fl = this.factor(day, "longer", q), fw = this.factor(day, "work", q);
    const intercitySummary = r.intercitySummary.slice();
    const ip = r.intercityParts;
    // T-276: longer trips by their factor, commutes between zones by work's
    if (ip && ip.length >= 4) { intercitySummary[1] = ip[0] * fl + ip[2] * fw; intercitySummary[2] = ip[1] * fl + ip[3] * fw; }
    else { intercitySummary[1] *= fl; intercitySummary[2] *= fl; }
    const intercityParts = ip?.map((v, i) => v * (i < 2 ? fl : fw));
    const corridors = r.corridors.slice();
    for (let i = 0; i + 3 < corridors.length; i += 4) { corridors[i + 2] *= fl; corridors[i + 3] *= fl; }
    const commuteCorridors = r.commuteCorridors?.map((v) => v * fw);
    const elsewhere = r.elsewhere?.map((v) => v * fl);
    const summary = r.summary.slice();
    summary[1] = summary[2] = summary[3] = 0;
    for (const v of [...Object.values(citySummary), intercitySummary]) for (const k of [1, 2, 3]) summary[k] += v[k] ?? 0;
    return { ...r, seg, board, alight, loadOfCrush, citySummary, intercitySummary, intercityParts, corridors, commuteCorridors, elsewhere, summary, parts: { keys: p.keys, seg: ps, board: pb, alight: pa } };
  }

  /** The views of one kind of day from its period results. */
  private dayDemand(s: Solve, round: number, final: boolean, day: DemandDay, res: Result[]): DayDemand {
    let trips = 0, rail = 0, walk = 0;
    const byPeriod: [string, number][] = [];
    res.forEach((r, q) => {
      const [, t, ra, wa] = r.summary;
      trips += t;
      rail += ra;
      walk += wa;
      byPeriod.push([PERIODS[q].name, t > 0 ? (100 * ra) / t : 0]);
    });
    const lines = new Map<string, LineDemand>();
    const stations = new Map<string, StationDemand>();
    const st = (id: string) => {
      let v = stations.get(id);
      if (!v) stations.set(id, (v = { boardings: 0, alightings: 0 }));
      return v;
    };
    for (const { id, o, n, stops } of s.layout) {
      const boardings = stops.map(() => 0);
      let fullest = 0;
      const loads: Float32Array[] = [], board: Float32Array[] = [], alight: Float32Array[] = [];
      res.forEach((r, q) => {
        for (let k = 0; k < n; k++) {
          // stop k is route node o + k on run 0 and o + n + (n - 1 - k) on run 1
          const a = o + k, b = o + 2 * n - 1 - k;
          boardings[k] += r.board[a] + r.board[b];
          const sd = st(stops[k]);
          sd.boardings += r.board[a] + r.board[b];
          sd.alightings += r.alight[a] + r.alight[b];
        }
        const seg = new Float32Array(2 * (n - 1));
        for (let k = 0; k < n - 1; k++) {
          seg[k] = r.seg[o + k];
          seg[n - 1 + k] = r.seg[o + n + k];
        }
        loads.push(seg);
        // the boarding account's input (T-164): boardings and alightings at each route node
        board.push(r.board.slice(o, o + 2 * n));
        alight.push(r.alight.slice(o, o + 2 * n));
        if (PERIODS[q].demand === "high")
          for (let k = 0; k < 2 * n; k++) fullest = Math.max(fullest, r.loadOfCrush[o + k]);
      });
      lines.set(id, { ridersPerDay: boardings.reduce((x, y) => x + y, 0), fullestPct: Math.round(100 * fullest), boardings, loads, board, alight });
    }
    const pct = (v: number) => (trips > 0 ? (100 * v) / trips : 0);
    const network: DemandView = {
      version: s.version,
      day,
      round,
      final,
      split: { train: pct(rail), walk: pct(walk), drive: Math.max(0, 100 - pct(rail) - pct(walk)) },
      trainShareByPeriod: byPeriod,
      peakHourFactor: PERIODS.map((_, q) => this.periodTable[q * PERIOD_STRIDE + 5] ?? 1),
      tripsPerDay: trips,
      railTripsPerDay: rail,
      lines,
      stations,
      purposes: purposeDemand(s.layout, res),
    };
    // only the cities this solve covered (T-149: a city not loaded has no view, and its controls wait)
    const solved = CITIES.filter(({id}) => res.every((r) => r.citySummary[id]));
    const cities = new Map<CityId, DemandView>(solved.map(({id}) => {
      let trips = 0, rail = 0, walk = 0;
      const trainShareByPeriod: [string, number][] = res.map((r, q) => {
        const [, t, ra, wa] = r.citySummary[id]; trips += t; rail += ra; walk += wa;
        return [PERIODS[q].name, t > 0 ? 100 * ra / t : 0];
      });
      const pct = (n: number) => trips > 0 ? 100 * n / trips : 0;
      return [id, {...network, tripsPerDay: trips, railTripsPerDay: rail, trainShareByPeriod, split: {train: pct(rail), walk: pct(walk), drive: Math.max(0, 100 - pct(rail) - pct(walk))}}];
    }));
    // T-325: and the zones open bare, from their own figures
    for (const id of this.bareIds()) if (!cities.has(id)) cities.set(id, this.bareView(id, day, network));
    // the regional layer between places: longer trips and (T-276) commutes between zones together
    const corridors = new Map<string, {from: number; to: number; trips: number; rail: number; commutes: number; commuteRail: number; air: number; car: number}>();
    let interTrips = 0, interRail = 0, commuteTrips = 0, commuteRail = 0;
    const elsewhere = INTERCITY_PLACES.map(() => 0);
    for (const r of res) {
      interTrips += r.intercitySummary[1]; interRail += r.intercitySummary[2];
      if (r.intercityParts && r.intercityParts.length >= 4) { commuteTrips += r.intercityParts[2]; commuteRail += r.intercityParts[3]; }
      if (r.corridorModes?.length) this.corridorModes = r.corridorModes;
      r.elsewhere?.forEach((v, p) => { if (p < elsewhere.length) elsewhere[p] += v; });
      for (let i = 0, k = 0; i < r.corridors.length; i += 4, k += 2) {
        const [from, to, trips, rail] = r.corridors.slice(i, i + 4), key = `${from}:${to}`;
        const ct = r.commuteCorridors?.[k] ?? 0, cr = r.commuteCorridors?.[k + 1] ?? 0;
        const old = corridors.get(key) ?? {from, to, trips: 0, rail: 0, commutes: 0, commuteRail: 0, air: 0, car: 0};
        // T-378: the longer trips not by rail split by the corridor's shares; commutes never fly
        const [air, car] = corridorSplit(this.corridorModes, i / 4, trips - rail);
        old.trips += trips + ct; old.rail += rail + cr; old.commutes += ct; old.commuteRail += cr;
        old.air += air; old.car += car + ct - cr;
        corridors.set(key, old);
      }
    }
    return { view: network, cities, intercity: {version: s.version, round, tripsPerDay: interTrips, railTripsPerDay: interRail, commuteTripsPerDay: commuteTrips, commuteRailPerDay: commuteRail, corridors: [...corridors.values()], elsewhere} };
  }
}

/** The purposes in `day_factors()`'s order (sim/src/demand/days.rs): work, the everyday layer's
 * kinds (`EVERYDAY_KEYS`), the market's longer trips. */
const PURPOSE_INDEX = Object.fromEntries((["work", ...EVERYDAY_KEYS, "longer"] as PurposeKey[]).map((k, i) => [k, i])) as Record<PurposeKey, number>;
const PURPOSE_INDEX_COUNT = 2 + EVERYDAY_KEYS.length;

let client: DemandClient | null = null;

const TEST = new URLSearchParams(location.search).get("demandTest") === "1";
/** T-351: the page starts a new game (`?new=1`, workers/clockClient.ts): no stations, so the viewed
 * zone opens bare from the start. */
const NEW_GAME = !TEST && !perfOff("bareStart") && new URLSearchParams(location.search).get("new") === "1";

/** Start the pool for a city. With `?demandTest=1` it solves T-005's hand-made network. */
export function startDemand(city = HOME_CITY) {
  if (client) return client;
  const c = (client = new DemandClient(city));
  setTripQuery((from, to) => c.trip(from, to)); // T-195
  setDemandQueries({ transfers: (members) => c.transfers(members), cityCells: (city) => c.cityCells(city), cellModes: (purpose, city) => c.purposeModes(purpose ?? "all", city), stationRiders: (id, top, members) => c.stationRiders(id, top, members), flows: (end, cells) => c.flows(end, cells), intercitySelection: (end, cells) => c.intercitySelection(end, cells), intercityStation: (id, members) => c.intercityStation(id, members) });
  if (TEST) c.submitNetwork(demandTestSnapshot());
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twDemand = { client: c, log: c.log, days: () => demandByDay.peek(), start: c.start, resubmit: () => c.resubmit(), view: () => demandView.peek(), cities: () => cityDemandViews.peek(), nearby: () => nearbyZones.peek(), intercity: () => intercityDemand.peek(), activeCity, cellModes: () => c.cellModes(), stationRiders: (id: string) => c.stationRiders(id), cityCells: () => c.cityCells(), flows: (end: "home" | "work", cells: Uint32Array) => c.flows(end, cells), intercitySelection: (end: "home" | "work", cells: Uint32Array) => c.intercitySelection(end, cells), intercityStation: (id: string) => c.intercityStation(id), transfers: (members: string[]) => c.transfers(members), trip: (from: [number, number], to: [number, number]) => c.trip(from, to) };
  return c;
}

/** Hand the clock worker's NetworkSnapshot to the pool (from clockClient's onNetworkSnapshot). */
export function submitNetwork(snap: NetworkSnapshot) {
  if (!TEST) client?.submitNetwork(snap);
}
