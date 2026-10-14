// Main-thread side of the shared-world demand pool (T-116/T-117).
//
// - Loads only the packs a game needs (T-149): the regional market, the viewed city first, and
//   every city a built station can reach (workers/cityReach.ts); a city viewed later loads then.
//   A solve waits until every city it needs is open in every worker, so no constructed city's
//   riders are ever left out of a published result. Each worker owns its copy of every
//   population; the pool budget applies to the whole world, not each city.
// - T-325: a zone no station reaches, viewed or shown beside the view, opens `bare` in the aux
//   worker alone, once the network's reach is known (until then the viewed city opens in every
//   worker, as before). Its city figures and bubbles come from there (no rail: walking and
//   driving only), no solve covers it and none is started for it. A station built within reach
//   opens it in every worker, and the solves cover it from then on.
// - Reports what it is loading (game/loading.ts `loadProgress`) and releases a loaded game's
//   clock hold once that game's network has its first estimate (READY_ROUND, T-172).
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
import { activeCity, CITIES, getWorldFrame, HOME_CITY, LEGACY_FRAME, MARKET_PACK, marketToWorld, viewedZone, worldFrameEpoch, type CityId } from "../game/cities";
import type { FrameSpec } from "../game/frame";
import { DAY_KINDS, type DayKind } from "../game/days";
import { shownDay } from "../game/ui";
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
} from "../game/demand";
import { makeFrame } from "../game/frame";
import { demandIdle, loadComplete, loadFailed, loadProgress, startup, takeGameLoadHold, zoneLoads } from "../game/loading";
import { PERIODS, periodsMatch } from "../game/periods";
import { EVERYDAY_KEYS, membersOf, PURPOSES, shownKey, type PurposeFilter, type PurposeKey } from "../game/purposes";
import { perfOff } from "../perfFlags";
import { citiesReached, loadFootprints } from "./cityReach";
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
 * A loaded game's demand is ready, and its clock runs, once this round is published: the
 * free-flow estimate (T-172). The crowding rounds then refine it in the background, as after
 * any edit. (T-149 waited for the last round; on the real New York save that was 8 s more.)
 */
const READY_ROUND = 0;
const PERIOD_STRIDE = 6;

/**
 * Workers in the pool: half the logical cores less one, at most 5 (one per period), at least 1.
 * On the real combined network one period's stage takes over a second, so three workers (two
 * periods on two of them) take twice as long to a result as five (T-119); a 16-thread desktop gets
 * five, an 8-thread laptop three, a 4-thread one a single worker. `?demandWorkers=N` (1-5)
 * overrides it.
 */
export function poolSize(): number {
  const q = Number(new URLSearchParams(location.search).get("demandWorkers"));
  if (Number.isInteger(q) && q >= 1 && q <= PERIODS.length) return q;
  const hc = navigator.hardwareConcurrency || 4;
  return Math.max(1, Math.min(PERIODS.length, Math.floor(hc / 2) - 1));
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
const ALL_PERIODS = (1 << PERIODS.length) - 1;
const frameOf = (): FrameSpec => { const f = getWorldFrame(); return { kind: f.kind, lon: f.lon, lat: f.lat }; };
/** A point the regional market reports (in its own pack frame) as a world point. */
const marketXY = (x: number, y: number): { x: number; y: number } => marketToWorld(x, y);
/** A place's centre (lon, lat) in world metres. */
const centreXY = ([lon, lat]: [number, number]): { x: number; y: number } => { const [x, y] = getWorldFrame().fromLonLat(lon, lat); return { x, y }; };
/** The zone count of a longer-trip answer of `len` numbers. For M markets a selection has 1+8*M
 * header numbers, 4 per zone of flows, 3 per zone of x, y, market, then (T-276) the commutes
 * between zones: 4 per market and 4 per zone; a station has 6*M header numbers then 5 per zone. */
function intercityZones(kind: "icSel" | "icStation", len: number): number {
  const m = INTERCITY_PLACES.length;
  return kind === "icSel" ? (len - 1 - 12 * m) / 11 : (len - 6 * m) / 5;
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
  /** T-232: its estimates before the final round are shown (startup, a game load, a city with no
   * figures yet); otherwise the last final result stays shown until this one's final round */
  showEstimates: boolean;
}

/** When one item (a city or the market) was asked for, downloaded and open in every worker:
 * performance.now() (window.twDemand.start.loads with ?debug=1). */
interface LoadLog { item: string; requestedAt: number; downloadedAt?: number; openedAt?: number; bare?: boolean }
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
const OPEN_MS_PER_MB = { city: 130, market: 40 };
const SOLVE_MS = { base: 600, perStop: 2.2 };
const STEP_CAP = 0.98;
/** a pack whose size is not known yet (its download has not started): about a mid-sized city */
const GUESS_BYTES = 6e6;
const LAST_SOLVE = "anitabuilder-load-solve";
const MARKET = "market";

const solveModel = (stops: number) => SOLVE_MS.base + SOLVE_MS.perStop * stops;
function lastSolve(): { ms: number; stops: number } | null {
  try {
    const v = JSON.parse(localStorage.getItem(LAST_SOLVE) || "null");
    return v && v.ms > 0 && v.stops >= 0 ? v : null;
  } catch { return null; }
}
/** Expected ms from a solve's start to its first estimate; `stops` null = the network is not known yet. */
function solveEstimate(stops: number | null): number {
  const last = lastSolve();
  const n = stops ?? last?.stops ?? 0;
  return last ? last.ms * solveModel(n) / solveModel(last.stops) : solveModel(n);
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
  private modesP: { version: number; round: number; day: DayKind; p: Promise<CellModes | null> } | null = null;
  /** T-260 `day_factors()` from the first worker: load factor per kind of day, purpose and
   * period, then each day's trip rate per purpose (sim/src/demand/days.rs) */
  private dayTable: number[] = [];
  /** `purposeModes` answers: `version:round:purpose` */
  private purposeP = new Map<string, Promise<CellModes | null>>();
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
  private bareOpen = new Map<CityId, BareFigures>();
  /** bytes of each item, the weight of its opening in the percentage */
  private bytes = new Map<string, number>();
  /** footprints for `needed`; "failed": every city counts as reached */
  private footprints: Awaited<ReturnType<typeof loadFootprints>> | "failed" | null = null;
  private reach = new WeakMap<Service, Set<CityId>>();
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
  /** resolves once the zone footprints are known (or failed) */
  private footprintsReady: Promise<unknown> = Promise.resolve();
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
    for (let i = 0; i <= n; i++) {
      const w = new Worker(new URL("./demand.worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<FromDemand>) => this.receive(i, e.data);
      w.onerror = (e) => this.fail(`worker ${i}: ${e.message}`);
      this.workers.push(w);
      this.periodsOf.push(i < n ? PERIODS.map((_, q) => q).filter((q) => q % n === i) : []);
      // the WASM starts at once; packs follow as they arrive (T-149)
      // T-325: the aux worker is told every solving worker's periods, for the zones it holds bare
      const msg: ToDemand = { kind: "init", packs: [], periods: this.periodsOf[i], aux: i === n, warmEveryday: i === 0 && n > 1 && !perfOff("stagger"), ...(i === n ? { groups: this.periodsOf.slice(0, n) } : {}) };
      w.postMessage(msg);
    }
    this.footprintsReady = loadFootprints().then((f) => { this.footprints = f; }, (err) => {
      console.warn("demand: city footprints unavailable, loading every city", err);
      this.footprints = "failed";
    });
    this.footprintsReady.then(() => this.ensure());
    // the viewed city loads when it changes (a pan, the city buttons)
    effect(() => { activeCity.value; viewedZone.value; this.ensure(); });
    // and the zones beside it with bubbles on screen (T-276), once nothing waits on them
    effect(() => { nearbyZones.value; this.ensure(); });
  }

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
   * footprints still loading). */
  private reachedSet(svc: Service | null): Set<CityId> | null {
    if (!svc) return null;
    if (svc.stations.length === 0) return new Set();
    if (this.footprints === "failed") return new Set(CITIES.map((c) => c.id));
    if (!this.footprints) return null;
    let reached = this.reach.get(svc);
    if (!reached) {
      const f = makeFrame(svc.frame), ll = new Float64Array(2 * svc.stations.length);
      svc.stations.forEach((s, k) => { const [lon, lat] = f.toLonLat(s.x, s.y); ll[2 * k] = lon; ll[2 * k + 1] = lat; });
      this.reach.set(svc, (reached = citiesReached(this.footprints, ll)));
    }
    return reached;
  }

  /** T-325: the cities a solve of `svc` must cover: every one its stations reach, and every one
   * opened in all workers (the viewed city at startup, before the reach was known). Null while
   * the reach is not known. */
  private solveNeed(svc: Service | null): Set<CityId> | null {
    const reached = this.reachedSet(svc);
    if (!reached) return null;
    const out = new Set(reached);
    for (const id of this.fetching) if (id !== MARKET) out.add(id as CityId);
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

  /** Every city in `need` and the market are open in every worker. */
  private allOpen(need: Set<CityId>): boolean {
    return this.ready === this.workers.length && this.opened.has(MARKET) && [...need].every((id) => this.opened.has(id));
  }

  /** Fetch whatever the newest service and the view need, the viewed city first; then solve if all is open. */
  private ensure() {
    if (this.status.startsWith("failed")) return;
    const known = this.needed(this.latest());
    const need = known ?? this.viewedSet();
    demandIdle.value = known?.size === 0; // T-318
    // over dead space with no station in reach there is nothing to open, the market included
    if (need.size && !this.fetching.has(MARKET)) this.fetchMarket();
    // T-325: a city no station reaches opens bare; while the reach is not known (startup), the
    // viewed city opens in every worker as before
    const reached = this.reachedSet(this.latest());
    for (const id of need) this.want(id, reached === null || reached.has(id));
    if (this.nearbyOpen()) for (const id of nearbyZones.peek()) this.want(id, !!reached?.has(id));
    this.flush();
    this.progress();
  }

  /** Fetch `id` to open in every worker (`full`), or bare in the aux worker, unless already. */
  private want(id: CityId, full: boolean) {
    if (full) { if (!this.fetching.has(id)) this.fetchCity(id); }
    else if (!this.fetching.has(id) && !this.fetchingBare.has(id)) this.fetchCity(id, true);
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
  private post(open: { pack?: { id: CityId; header: string; bin: ArrayBuffer }; intercity?: Uint8Array }) {
    this.workers.forEach((w, i) => {
      const last = i === this.workers.length - 1;
      const pack = open.pack && { ...open.pack, bin: last ? open.pack.bin : open.pack.bin.slice(0) };
      // T-180: the market as bytes, each solving worker its own copy (the last solving one the
      // original), decoded there: decoding 8 MB and cloning the text six times held the main thread
      // up while the player could already pan. The aux worker never opens it: an empty marker.
      const market = !open.intercity ? undefined : i === this.computeN ? new Uint8Array(0) : i === this.computeN - 1 ? open.intercity : open.intercity.slice();
      const msg: ToDemand = { kind: "open", packs: pack ? [pack] : [], intercity: market };
      const transfer: Transferable[] = pack ? [pack.bin] : [];
      if (market?.byteLength) transfer.push(market.buffer as ArrayBuffer);
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

  /** `bare` (T-325): to the aux worker alone. A pack opened bare and reached later is fetched again
   * for every worker (the page keeps no copy). */
  private fetchCity(id: CityId, bare = false) {
    (bare ? this.fetchingBare : this.fetching).add(id);
    const log: LoadLog = { item: id, requestedAt: performance.now(), ...(bare ? { bare } : {}) };
    this.start.loads.push(log);
    Promise.all([this.download(`${id}.json`), this.download(`${id}.bin`)]).then(([header, bin]) => {
      log.downloadedAt = performance.now();
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

  private fetchMarket() {
    this.fetching.add(MARKET);
    const log: LoadLog = { item: MARKET, requestedAt: performance.now() };
    this.start.loads.push(log);
    this.download(`${MARKET_PACK}.json`).then((text) => {
      log.downloadedAt = performance.now();
      this.bytes.set(MARKET, text.byteLength);
      this.hashPack(MARKET, [text]);
      this.post({ intercity: text });
    }).catch((err) => this.fail(`regional market: ${String(err)}`));
  }

  /** A worker opened `item`: once all have, waiting solves and queries may go. */
  private openedBy(item: string) {
    const n = (this.acks.get(item) ?? 0) + 1;
    this.acks.set(item, n);
    if (n < this.workers.length) return;
    this.opened.add(item);
    const log = this.start.loads.find((l) => l.item === item && !l.bare);
    if (log) log.openedAt = performance.now();
    this.settleWaiters();
  }

  /** T-325: the aux worker opened zones bare: their figures and city views. */
  private bareOpened(figures: Record<CityId, BareFigures>) {
    for (const [id, f] of Object.entries(figures)) {
      this.bareOpen.set(id, f);
      const log = this.start.loads.find((l) => l.item === id && l.bare);
      if (log) log.openedAt = performance.now();
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
    if ([...need].some((id) => !s.cities.includes(id))) this.dispatch(s.svc, performance.now());
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

  /** T-343: every zone fetched (in every worker or bare) whose figures have not arrived, with its
   * share done (game/loading.ts `zoneLoads`): the load in hand's for its zones, else its own
   * download and opening, each weighted by its expected time as the load's are. */
  private zoneProgress() {
    const shown = cityDemandViews.peek(), now = performance.now();
    const next = new Map<CityId, number>();
    let moving = false;
    for (const id of [...this.fetching, ...this.fetchingBare]) {
      if (id === MARKET || shown.has(id) || next.has(id) || this.status.startsWith("failed")) continue;
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
        const log = this.start.loads.find((l) => l.item === id && l.downloadedAt !== undefined && !l.openedAt);
        const open = this.isOpen(id) ? 1 : log ? Math.min(STEP_CAP, (now - log.downloadedAt!) / openMs) : 0;
        f = (dlMs * dl + openMs * open) / (dlMs + openMs);
        if (!this.isOpen(id)) moving = true;
      }
      const best = Math.max(this.zoneBest.get(id) ?? 0, Math.min(0.99, f));
      next.set(id, best);
    }
    this.zoneBest = new Map(next);
    const prev = zoneLoads.peek();
    if (next.size !== prev.size || [...next].some(([id, f]) => prev.get(id) !== f)) zoneLoads.value = next;
    // a bare zone's load moves with time while it downloads and opens; nothing else ticks for it
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
    const loaded = (item: string) => (item === MARKET || sn === null || sn.has(item as CityId) ? this.opened.has(item) : this.isOpen(item));
    // T-328: a zone the shown cached result covers has its figures; it loads unseen
    const cached = this.cacheShown() ? this.cachedCities : null;
    const missing = [...need].filter((id) => !loaded(id) && !cached?.has(id)).sort((a, b) => (a === viewed ? -1 : b === viewed ? 1 : 0));
    const atStartup = startup.peek() === "loading";
    const s = this.cur;
    if (!this.job) {
      // T-243: a loaded game whose cities are all open (a browser slot) still waits for its solve
      const heldSolve = !!this.hold && this.hold.version !== undefined && s?.version === this.hold.version && s.published < READY_ROUND;
      if (!missing.length && !(atStartup && !this.opened.has(MARKET)) && !heldSolve) {
        if (!atStartup) loadProgress.value = null;
        return;
      }
      this.job = { items: [], cities: [], startup: atStartup, best: 0, version: heldSolve && !missing.length ? s!.version : undefined };
    }
    const job = this.job;
    for (const id of missing) if (!job.items.includes(id)) { job.items.push(id); job.cities.push(id); }
    if (!this.opened.has(MARKET) && !job.items.includes(MARKET)) job.items.push(MARKET);
    if (atStartup && need.has(viewed) && !job.cities.includes(viewed)) job.cities.unshift(viewed);
    const needsSolve = this.jobNeedsSolve(job);
    const solved = !needsSolve || (!!s && job.version !== undefined && s.version === job.version && s.published >= READY_ROUND);
    if (solved && job.items.every(loaded)) {
      // remembered for the next load's estimate (T-243): a startup's solve, so it describes the
      // game a reload opens
      if (job.startup && s && s.firstAt !== undefined) {
        const stops = s.svc.lines.reduce((n, l) => n + l.stops.length, 0);
        try { localStorage.setItem(LAST_SOLVE, JSON.stringify({ ms: Math.round(s.firstAt - s.sentAt), stops })); } catch { /* storage unavailable */ }
      }
      this.job = null;
      clearInterval(this.progressTick);
      this.progressTick = 0;
      loadProgress.value = null;
      return;
    }
    if (!this.progressTick) this.progressTick = setInterval(() => this.progress(), 100) as unknown as number;
    const now = performance.now();
    // time against the expectation: in step with it to 90%, then slowing toward STEP_CAP, so a
    // step that runs late still creeps rather than stopping dead
    const within = (since: number | undefined, expected: number) => {
      if (since === undefined) return 0;
      const x = Math.max(0, now - since) / Math.max(1, expected);
      return x < 0.9 ? x : 0.9 + (STEP_CAP - 0.9) * (1 - Math.exp(-(x - 0.9) / 0.3));
    };
    // Downloading: bytes; expected to take as long as its rate so far says.
    let got = 0, total = 0, dlStart = Infinity;
    // Opening: each item in every worker, expected from its size; worker acks are its real progress.
    let openMs = 0, openDone = 0, openStart = Infinity;
    for (const item of job.items) {
      let size = 0;
      for (const f of item === MARKET ? [`${MARKET_PACK}.json`] : [`${item}.json`, `${item}.bin`]) {
        const d = this.downloads.get(f);
        if (d) { got += Math.min(d.got, d.total || d.got); total += d.total || d.got; dlStart = Math.min(dlStart, d.at); size += d.total || d.got; }
      }
      const ms = (this.bytes.get(item) ?? (size || GUESS_BYTES)) / 1e6 * (item === MARKET ? OPEN_MS_PER_MB.market : OPEN_MS_PER_MB.city);
      openMs += ms;
      openDone += loaded(item) ? ms : ms * (this.acks.get(item) ?? 0) / this.workers.length;
      const log = this.start.loads.find((l) => l.item === item);
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
    const solveMs = needsSolve ? solveEstimate(svc ? svc.lines.reduce((n, l) => n + l.stops.length, 0) : null) : 0;
    let solve = 0;
    if (s && job.version !== undefined && s.version === job.version) {
      let landed = 0;
      for (let r = 0; r <= READY_ROUND; r++) landed += r <= s.published ? PERIODS.length : s.byRound[r].filter(Boolean).length;
      solve = Math.max(landed / (PERIODS.length * (READY_ROUND + 1)), within(s.sentAt, solveMs));
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
    loadProgress.value = { cities: job.cities.filter((id) => id !== MARKET), fraction: job.best, step, startup: job.startup };
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

  /** The cache key of a solve covering `cities`, once their packs and the market have downloaded. */
  private async keyFor(cities: CityId[]): Promise<CacheKey> {
    const ids = [MARKET, ...cities];
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
      await this.footprintsReady;
      const need = this.solveNeed(svc);
      const cities = Object.keys(stored.packs ?? {}).filter((k) => k !== MARKET);
      if (!need || need.size !== cities.length || cities.some((id) => !need.has(id))) return void (log.miss = "zones");
      const why = sameKey(stored, await this.keyFor(cities));
      if (why) return void (log.miss = why);
      if (this.latest() !== svc || (this.cur?.svc === svc && this.cur.published >= 0) || this.status.startsWith("failed")) return void (log.miss = "late");
      this.applyCache(svc, canon, stored, hash, cities);
      log.shownAt = performance.now();
    } catch (err) {
      log.miss = "error: " + String(err);
    }
  }

  /** Show a cached result as a final one under CACHE_VERSION. The clock runs on it; the solve of the
   * same network carries on, its estimates are not shown, and its final round replaces it whole. */
  private applyCache(svc: Service, ids: ReturnType<typeof canonical>, stored: StoredDemand, hash: string, cities: CityId[]) {
    const days = fromStored(stored, ids);
    if (days.length !== DAY_KINDS.length) throw new Error("days");
    const now = performance.now();
    // a solve entry for it, so its layout and zones are known as a shown solve's are (T-324, T-325);
    // first in the log, so the newest solve stays last
    this.log.unshift({ version: CACHE_VERSION, svc, layout: packService(svc).layout, results: [], byRound: [], published: stored.round, arrivedAt: now, sentAt: now, periodMs: [], setupMs: [], memBytes: [], cities, showEstimates: false });
    this.cachedCities = new Set(cities);
    this.cachedHash = hash;
    this.cachedModes = stored.modes ?? null;
    if (this.cur?.svc === svc) this.cur.showEstimates = false;
    demandByDay.value = days;
    this.addBareViews();
    demandUpdating.value = true; // "Updating demand" until the fresh final
    // the load in hand is covered; the clock runs
    this.job = null;
    clearInterval(this.progressTick);
    this.progressTick = 0;
    loadProgress.value = null;
    if (this.hold) {
      this.hold = null;
      loadComplete();
      if (nearbyZones.peek().length) this.ensure();
    }
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
    const [hash, key] = await Promise.all([sha256(canon.text), this.keyFor(s.cities)]);
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

  /** A held game's network has its first estimate: the clock runs. */
  private checkHold() {
    if (this.hold && this.idle()) return this.settleIdle(); // T-318
    const s = this.cur;
    // (and the viewed zone open, bare perhaps, so its figures show: T-325)
    if (this.hold && s && this.hold.version === s.version && s.published >= READY_ROUND && !this.waiting && [...this.viewedSet()].every((id) => this.isOpen(id))) {
      this.hold = null;
      loadComplete();
      // the nearby zones' packs may load now (T-276)
      if (nearbyZones.peek().length) this.ensure();
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
      if (msg.market) this.openedBy(MARKET);
      this.flush();
      this.progress();
      return;
    }
    const s = this.cur;
    if (!s || msg.version !== s.version) return; // superseded
    if (msg.kind === "done") return;
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
    const showEstimates = !demandView.peek()?.final || !!this.hold || cities.some((id) => !shown.has(id));
    this.cur = { version, svc, layout, results: PERIODS.map(() => null), byRound: Array.from({length: ROUNDS + 1}, () => PERIODS.map(() => null)), published: -1, arrivedAt, sentAt: performance.now(), periodMs: [], setupMs: [], memBytes: [], cities, showEstimates };
    this.log.push(this.cur);
    if (this.log.length > 50) this.log.shift();
    demandUpdating.value = true;
    demandStep.value = { done: 0, of: ROUNDS + 1 };
    const msg: ToDemand = { kind: "solve", version, net, rounds: ROUNDS };
    for (const w of this.workers) w.postMessage(msg);
    this.progress(); // T-243: a held game's solve shows its progress even with nothing to load
  }

  // --- Demand views (T-078; types and meaning in game/demand.ts) ---------------------------------

  /** Send `make(id)` to the given workers (all by default) and collect their replies. */
  private ask(make: (id: number) => Query, which: number[] = this.workers.slice(0, this.computeN).map((_, i) => i), transfer: Transferable[] = [], zone?: CityId): Promise<Reply[]> {
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
      for (const i of which) this.workers[i].postMessage({...q, city, round: demandView.peek()?.round ?? -1, day: shownDay.peek()}, transfer);
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
    const key = `${version}:${round}:${day}:${purpose}`;
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
        for (const r of byMask(await this.ask((id) => ({ kind: "purposeModes", id, version, purpose: kind }), undefined, [], other))) {
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
    if (everydayToo && other === undefined && this.modesP?.version === version && this.modesP.round === round && this.modesP.day === day) return this.modesP.p;
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
      const sent = day === 0 ? this.pushed.get(`${version}:${round}:${city}`) : undefined;
      const parts: { mask: number; sums: Float32Array | null; everyday: Float32Array | null }[] = sent && sent.length === this.computeN ? sent : (await this.ask((id) => ({ kind: "subSums", id, version }), undefined, [], other)).map((r) => (r.kind === "subSums" ? r : { mask: 0, sums: null, everyday: null }));
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
      const [r] = await this.ask((id) => ({ kind: "cellModes", id, version, sums: sums! }), [this.computeN], [sums.buffer], other);
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
    this.modesP = { version, round, day, p };
    p.then((v) => {
      if (!v && this.modesP?.p === p) this.modesP = null; // not cached when it failed
    });
    return p;
  }

  /** T-325: `cellModes` of a zone open bare, from the aux worker's no-rail pass: the commuters by
   * mode (no rail) and each solving worker's periods' everyday legs, added up as `cellModes` adds
   * the workers' answers for a solved city. */
  private async bareModes(city: CityId, everydayToo: boolean, version: number, round: number, day: DayKind, other: CityId | undefined): Promise<CellModes | null> {
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
  async stationRiders(stationId: string, top = 20, members: string[] = [stationId]): Promise<StationRiders | null> {
    const version = demandView.peek()?.version;
    const solve = this.log.find((s) => s.version === version);
    if (version === undefined || !solve) return null;
    const indices = members.map((m) => solve.svc.stations.findIndex((s) => s.id === m)).filter((i) => i >= 0);
    if (indices.length === 0) return null;
    const answers = await Promise.all(indices.map((station) => this.ask((id) => ({ kind: "station", id, version, station }))));
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
    for (const p of kept) {
      const nh = p[0], nw = p[1];
      let o = 2;
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
    if (!toZone || !fromZone) return null;
    const cells = (m: Map<number, number>): CellRiders => {
      const t = topOf([...m.keys()], [...m.values()]);
      return { cells: t.keys, riders: t.vals, total: t.vals.reduce((a, b) => a + b, 0) };
    };
    const zones = (v: Float32Array): ZoneRiders => {
      const t = topOf(Uint32Array.from(v, (_, i) => i), v, top);
      return { zones: t.keys, riders: t.vals };
    };
    return { version, stationId, scope: members.join(","), home: cells(home), work: cells(work), homeRidersWorkIn: zones(toZone), workRidersLiveIn: zones(fromZone) };
  }

  /** A station complex's transfers a day (T-176; game/demand.ts `ComplexTransfers`): the route
   * nodes at its members are sent, every worker answers for its periods, and the pairs are added
   * up by station and line at each end (both runs of a line together). */
  async transfers(members: string[]): Promise<ComplexTransfers | null> {
    const version = demandView.peek()?.version, round = demandView.peek()?.round ?? 0;
    const solve = this.log.find((s) => s.version === version);
    if (version === undefined || !solve) return null;
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
    for (const r of byMask(await this.ask((id) => ({ kind: "transfers", id, version, nodes: ids })))) {
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
    if (mask !== ALL_PERIODS || demandView.peek()?.version !== version) return null;
    return { version, round, scope: members.join(","), pairs: [...sum.values()].sort((x, y) => y.riders - x.riders) };
  }

  /** The trip explorer (T-195; game/trip.ts): one trip from `from` to `to` (world metres) as a
   * commute, every period from the worker holding it. Null while there is no solve, or the
   * workers have moved on to a newer network or round. */
  async trip(from: [number, number], to: [number, number]): Promise<Trip | null> {
    const version = demandView.peek()?.version, round = demandView.peek()?.round ?? 0;
    const solve = this.log.find((s) => s.version === version);
    if (version === undefined || !solve) return null;
    const at: RouteNodes = [];
    for (const { id, o, n, stops } of solve.layout)
      for (let k = 0; k < n; k++) at[o + k] = at[o + 2 * n - 1 - k] = { station: stops[k], line: id, k };
    const periods: Trip["periods"] = PERIODS.map(() => null);
    let mask = 0, outside = false;
    for (const r of await this.ask((id) => ({ kind: "trip", id, version, from, to }))) {
      if (r.kind !== "trip" || !r.answers) return null;
      mask |= r.mask;
      r.periods.forEach((q, i) => {
        const v = r.answers![i];
        if (v[0] === -1) outside = true;
        periods[q] = readTripPeriod(v, at);
      });
    }
    if (mask !== ALL_PERIODS || demandView.peek()?.version !== version) return null;
    return { version, round, key: tripKey(from, to), periods, outside };
  }

  /** Where the commuters of a set of cells go (T-098; game/demand.ts `Flows`): every worker's rail
   * part of its periods, added up, then one worker spreads all modes over the far end. */
  async flows(end: "home" | "work", cells: Uint32Array): Promise<Flows | null> {
    const city = activeCity.peek(), round = demandView.peek()?.round;
    const version = demandView.peek()?.version;
    if (version === undefined || cells.length === 0) return null;
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
        this.ask((id) => ({ kind: "flowRail", id, version, end: e, cells })),
        this.ask((id) => ({ kind: "purposes", id, version, end: e, cells })),
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
      const [r] = await this.ask((id) => ({ kind: "flows", id, version, end: e, cells, sums: sums! }), [0], [sums.buffer]);
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
    if (activeCity.peek() !== city || demandView.peek()?.version !== version || demandView.peek()?.round !== round) return null;
    return result;
  }

  /** Add up every worker's answer to a longer-trip query (each holds some of the periods); null
   * unless all five periods answered at the version asked about. `matched` copies the first
   * answer's first number (the same on every worker). */
  private async sumIntercity(kind: "icSel" | "icStation", version: number, make: (id: number) => Query): Promise<Float64Array | null> {
    const parts = await this.ask(make);
    let mask = 0;
    let sums: Float64Array | null = null;
    // in worker (period) order, not arrival order: the same sums every time (T-146 compares them)
    for (const r of byMask(parts)) {
      if (r.kind !== kind || !r.packed) return null;
      mask |= r.mask;
      if (!sums) sums = r.packed.slice();
      else addIntercity(kind, sums, r.packed);
    }
    if (version !== demandView.peek()?.version || mask !== ALL_PERIODS) return null;
    return sums;
  }

  /** The longer trips of a bubble selection (game/demand.ts `IntercitySelection`, T-118). */
  async intercitySelection(end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null> {
    const version = demandView.peek()?.version;
    if (version === undefined || cells.length === 0) return null;
    const e = end === "home" ? 0 : 1;
    // T-336: a zone open bare: the solving workers hold the market but not the zone, so its squares
    // come from the aux worker with the question
    let sq: { xy: Float32Array; share: Float32Array } | undefined;
    if (this.isBare(activeCity.peek())) {
      const [r] = await this.ask((id) => ({ kind: "bareSquares", id, version, end: e, cells }), [this.computeN]);
      if (r?.kind !== "bareSquares" || !r.xy || !r.share) return null;
      sq = { xy: r.xy, share: r.share };
    }
    const v = await this.sumIntercity("icSel", version, (id) => ({ kind: "icSel", id, version, end: e, cells, ...(sq ? { sq } : {}) }));
    if (!v) return null;
    // per place, 8 numbers after the matched share: starts (all, rail, x, y) then ends; then per
    // far zone (all, rail) of starts and of ends, then zone x, y, market (not summed)
    const P = INTERCITY_PLACES.length;
    const head = 1 + 8 * P, nz = intercityZones("icSel", v.length), xy = intercityZoneXY("icSel", v.length);
    // T-276: the commutes between zones after the zones' x, y: per place, then per zone, the legs of
    // commuters living in the selection (by where they work) and of those working there (by where
    // they live), all and rail. Starts take the first, ends the second.
    const tail = xy + 3 * nz;
    const flows = (dir: number): PlaceFlows => {
      const places: PlaceTrips[] = [];
      let all = 0, rail = 0;
      for (let p = 0; p < P; p++) {
        const o = 1 + 8 * p + 4 * dir;
        const ca = v[tail + 4 * p + 2 * dir] ?? 0, cr = v[tail + 4 * p + 2 * dir + 1] ?? 0;
        const a = v[o] + ca, r = v[o + 1] + cr;
        all += a; rail += r;
        if (a > 0) {
          const city = CITIES.find((c) => c.market === p);
          places.push({ place: p, all: a, rail: r, ...(v[o] > 0 || !city ? marketXY(v[o + 2] / Math.max(v[o], 1e-9), v[o + 3] / Math.max(v[o], 1e-9)) : centreXY(city.center)) });
        }
      }
      const zones: ZoneTrips[] = [];
      for (let z = 0; z < nz; z++) {
        const c = tail + 4 * P + 4 * z + 2 * dir;
        const a = v[head + 4 * z + 2 * dir] + (v[c] ?? 0);
        if (a > 0) zones.push({ ...marketXY(v[xy + 3 * z], v[xy + 3 * z + 1]), place: v[xy + 3 * z + 2], all: a, rail: v[head + 4 * z + 2 * dir + 1] + (v[c + 1] ?? 0) });
      }
      return { all, rail, places: places.sort((a, b) => b.all - a.all), zones: zones.sort((a, b) => b.all - a.all) };
    };
    return { version, starts: flows(0), ends: flows(1), matched: v[0] };
  }

  /** A station's longer-trip rail riders (game/demand.ts `IntercityStation`, T-118), or a station
   * complex's (`members`, T-146), each demand station once as in `stationRiders`. */
  async intercityStation(stationId: string, members: string[] = [stationId]): Promise<IntercityStation | null> {
    const version = demandView.peek()?.version;
    const solve = this.log.find((s) => s.version === version);
    if (version === undefined || !solve) return null;
    const indices = members.map((m) => solve.svc.stations.findIndex((s) => s.id === m)).filter((i) => i >= 0);
    if (indices.length === 0) return null;
    const answers = await Promise.all(indices.map((station) => this.sumIntercity("icStation", version, (id) => ({ kind: "icStation", id, version, station }))));
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
        if (c > 0) places.push({ place: p, all: c, rail: c, ...marketXY(v[(2 + 2 * dir) * n + p] / c, v[(3 + 2 * dir) * n + p] / c) });
      }
      const zones: ZoneTrips[] = [];
      for (let z = 0; z < nz; z++) {
        const c = v[head + dir * nz + z];
        if (c > 0) zones.push({ ...marketXY(v[xy + 3 * z], v[xy + 3 * z + 1]), place: v[xy + 3 * z + 2], all: c, rail: c });
      }
      return { all: rail, rail, places: places.sort((a, b) => b.all - a.all), zones: zones.sort((a, b) => b.all - a.all) };
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
    demandByDay.value = DAY_KINDS.map((day) => this.dayDemand(s, round, final, day, day === 0 ? res : res.map((r, q) => this.dayResult(r, q, day))));
    if (final) {
      s.refinedAt = performance.now();
      if (s === this.cur) {
        demandUpdating.value = false;
        void this.writeCache(s); // T-328
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
      const day = k as DayKind;
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
   * T-325: a bare zone's view of one kind of day, worked out from its figures exactly as
   * `dayResult` and `dayDemand` work out a solved city's (the figures are what a solve gives it).
   */
  private bareView(id: CityId, day: DayKind, network: DemandView): DemandView {
    const b = this.bareOpen.get(id)!;
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

  /** T-325: zones opened bare since the shown result join its city views at once (no solve). */
  private addBareViews() {
    const days = demandByDay.peek();
    let changed = false;
    const next = days.map((d, k) => {
      if (!d.view) return d;
      const add = [...this.bareOpen.keys()].filter((id) => !d.cities.has(id));
      if (!add.length) return d;
      changed = true;
      const cities = new Map(d.cities);
      for (const id of add) cities.set(id, this.bareView(id, k as DayKind, d.view));
      return { ...d, cities };
    });
    if (changed) demandByDay.value = next;
  }

  /** A kind of day's load factor for purpose `key` in period `q` (sim/src/demand/days.rs). */
  private factor(day: DayKind, key: string, q: number): number {
    const p = PURPOSE_INDEX[key as PurposeKey] ?? 0;
    return this.dayTable[(day * PURPOSE_INDEX_COUNT + p) * PERIODS.length + q] ?? 1;
  }

  /**
   * One period's result as it is on another kind of day (T-260): each purpose's riders times its
   * load factor, the totals their sum, the mode split from each city's purposes the same way, the
   * market's trips and corridors by its own factor. The busiest-hour load of crush follows the
   * segment's riders. Route and mode choices are the weekday's (docs/demand.md 4.2).
   */
  private dayResult(r: Result, q: number, day: DayKind): Result {
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
    const summary = r.summary.slice();
    summary[1] = summary[2] = summary[3] = 0;
    for (const v of [...Object.values(citySummary), intercitySummary]) for (const k of [1, 2, 3]) summary[k] += v[k] ?? 0;
    return { ...r, seg, board, alight, loadOfCrush, citySummary, intercitySummary, intercityParts, corridors, commuteCorridors, summary, parts: { keys: p.keys, seg: ps, board: pb, alight: pa } };
  }

  /** The views of one kind of day from its period results. */
  private dayDemand(s: Solve, round: number, final: boolean, day: DayKind, res: Result[]): DayDemand {
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
    for (const id of this.bareOpen.keys()) if (!cities.has(id)) cities.set(id, this.bareView(id, day, network));
    // the regional layer between places: longer trips and (T-276) commutes between zones together
    const corridors = new Map<string, {from: number; to: number; trips: number; rail: number; commutes: number; commuteRail: number}>();
    let interTrips = 0, interRail = 0, commuteTrips = 0, commuteRail = 0;
    for (const r of res) {
      interTrips += r.intercitySummary[1]; interRail += r.intercitySummary[2];
      if (r.intercityParts && r.intercityParts.length >= 4) { commuteTrips += r.intercityParts[2]; commuteRail += r.intercityParts[3]; }
      for (let i = 0, k = 0; i < r.corridors.length; i += 4, k += 2) {
        const [from, to, trips, rail] = r.corridors.slice(i, i + 4), key = `${from}:${to}`;
        const ct = r.commuteCorridors?.[k] ?? 0, cr = r.commuteCorridors?.[k + 1] ?? 0;
        const old = corridors.get(key) ?? {from, to, trips: 0, rail: 0, commutes: 0, commuteRail: 0};
        old.trips += trips + ct; old.rail += rail + cr; old.commutes += ct; old.commuteRail += cr;
        corridors.set(key, old);
      }
    }
    return { view: network, cities, intercity: {version: s.version, round, tripsPerDay: interTrips, railTripsPerDay: interRail, commuteTripsPerDay: commuteTrips, commuteRailPerDay: commuteRail, corridors: [...corridors.values()]} };
  }
}

/** The purposes in `day_factors()`'s order (sim/src/demand/days.rs): work, the everyday layer's
 * kinds (`EVERYDAY_KEYS`), the market's longer trips. */
const PURPOSE_INDEX = Object.fromEntries((["work", ...EVERYDAY_KEYS, "longer"] as PurposeKey[]).map((k, i) => [k, i])) as Record<PurposeKey, number>;
const PURPOSE_INDEX_COUNT = 2 + EVERYDAY_KEYS.length;

let client: DemandClient | null = null;

const TEST = new URLSearchParams(location.search).get("demandTest") === "1";

/** Start the pool for a city. With `?demandTest=1` it solves T-005's hand-made network. */
export function startDemand(city = HOME_CITY) {
  if (client) return client;
  const c = (client = new DemandClient(city));
  setTripQuery((from, to) => c.trip(from, to)); // T-195
  setDemandQueries({ transfers: (members) => c.transfers(members), cityCells: (city) => c.cityCells(city), cellModes: (purpose, city) => c.purposeModes(purpose ?? "all", city), stationRiders: (id, top, members) => c.stationRiders(id, top, members), flows: (end, cells) => c.flows(end, cells), intercitySelection: (end, cells) => c.intercitySelection(end, cells), intercityStation: (id, members) => c.intercityStation(id, members) });
  if (TEST) c.submitNetwork(demandTestSnapshot());
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twDemand = { client: c, log: c.log, days: () => demandByDay.peek(), start: c.start, resubmit: () => c.resubmit(), view: () => demandView.peek(), cities: () => cityDemandViews.peek(), intercity: () => intercityDemand.peek(), activeCity, cellModes: () => c.cellModes(), stationRiders: (id: string) => c.stationRiders(id), cityCells: () => c.cityCells(), flows: (end: "home" | "work", cells: Uint32Array) => c.flows(end, cells), intercitySelection: (end: "home" | "work", cells: Uint32Array) => c.intercitySelection(end, cells), intercityStation: (id: string) => c.intercityStation(id), transfers: (members: string[]) => c.transfers(members), trip: (from: [number, number], to: [number, number]) => c.trip(from, to) };
  return c;
}

/** Hand the clock worker's NetworkSnapshot to the pool (from clockClient's onNetworkSnapshot). */
export function submitNetwork(snap: NetworkSnapshot) {
  if (!TEST) client?.submitNetwork(snap);
}
