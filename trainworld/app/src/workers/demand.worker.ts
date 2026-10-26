// Shared-world demand worker (T-116/T-117). Owns New York local commutes,
// Philadelphia local commutes and the separate regional market. Each assigned period
// advances all populations from one frozen total load vector at every crowding round.
// Network messages supersede work between populations; queries require the same
// version AND refinement round across all this worker's periods; one marked `wait` about a round
// not reached yet is answered when it is (T-327).
//
// Packs open as they arrive (T-149): `init` starts the WASM (and may carry packs), each `open`
// adds city packs or the regional market. A solve covers the cities open when it starts.
//
// One worker of the pool has no periods (`aux`, T-119): it only sets the network up. It is never
// busy, so the questions that need no period's result (the cells, a bubble's commuters by mode
// from the sums it is handed) are answered at once, not after a busy worker's current pass.
//
// T-325: the aux worker also holds the zones no station reaches, opened `bare`: their commute and
// everyday layers on an empty network, solved once for every period with no rail. Their bubbles
// and city figures come from there; no solve covers them, and no solving worker opens them.
//
// T-301: `close` frees packs (and the market) the page no longer needs; one the solve in hand
// covers goes when that solve ends.

import init, { DemandApi, EverydayApi, IntercityApi, MarketPacks, day_factors, demand_periods, everyday_warm_up, release_skims, wasm_memory_bytes } from "../wasm/demand/sim"; // demand only (T-051)
import { CITIES, cityById, cityToMarket, cityToWorld, HOME_CITY, lonLatToCity, marketWorld, setWorldFrame, widenBox, worldToCity, worldToLonLat, zoneBox, type CityId } from "../game/cities";
import { loadAttractions, type Attraction } from "../game/attractions";
import { EVERYDAY_KEYS, EVERYDAY_NAMES } from "../game/purposes";
import { PERIODS } from "../game/periods";
import { scaledKnots } from "../game/fareCurve";
import { FARE_BASELINE, fareFactors, fareScale, fareWeight } from "../game/income";
import type { CorridorTable } from "../game/demand";
import { TRIP_LAYOUT, type BareFigures, type DemandNetwork, type FromDemand, type MarketPack, type MarketRows, type OpenRejected, type ToDemand } from "./demandProtocol";
import { cellBox, cityReach, clusterView, reachView, splitClusters, type Box, type Cluster, type PopView } from "./demandClusters";

/** WASM started (and the market opened if `init` carried it): `open` and solves wait for this. */
let started = false;
/** init and open in arrival order, each after the one before (T-149: packs arrive one by one) */
let opening: Promise<void> = Promise.resolve();
let aux = false;
let warmed = false;
const cities = new Map<CityId, DemandApi>();
/** the cities the network in hand was set on (`current`'s job): only these answer queries */
let jobCities = new Set<CityId>();
const everyday = new Map<CityId, EverydayApi>();
const EVERYDAY_PURPOSES = EVERYDAY_NAMES;
/** T-377: the zones' market packs open here, and `intercity` the market composed of all of them
 * (composed again whenever one opens or closes); it answers every question's all-mode part */
let marketPacks: MarketPacks | null = null;
/** T-776: the world market's rows until `marketPacks` is made from them */
let marketRows: MarketRows | null = null;
let intercity: IntercityApi | null = null;
/** T-409: the market's parts by `Cluster.marketKey`, one per cluster the market reaches, each on its
 * cluster's network alone (`IntercityApi.beside`). A part keeps the composition it was made from
 * while its key lasts (its zones' packs are in it); a part whose key is gone is reused for a new
 * key when it is of the newest composition, the rest are freed */
const marketParts = new Map<string, IntercityApi>();
/** the market's open 6 km zones' world ids and boxes (`IntercityApi.zone_boxes`), which its
 * clusters are joined by */
let marketZones: Float64Array | null = null;
let mine: number[] = [];
let pending: { version: number; net: DemandNetwork; rounds: number; barrier?: boolean; whole?: boolean; oneMarket?: boolean; wide?: boolean; exactMarket?: boolean; fresh?: boolean; first?: CityId } | null = null;
let busy = false;
/** the version whose network `api` holds (queries about any other get nulls) */
let current = -1;
/** the round each of this worker's periods has reached; -1 before a result, -2 while a stage runs */
const roundsHeld: number[] = [];
/**
 * What `subSums` answers, taken the moment all this worker's periods have a round, per city and
 * round (T-119). The bubbles ask for it right after a round is published, when this worker is
 * already part way into the next one; the live state would be a mixture of two rounds by then, so
 * the answer is kept from when it was whole.
 */
const sumsByRound = new Map<string, { mask: number; sums: Float32Array; everyday: Float32Array | null }>();
/** the round the solve in hand is working on or last finished */
let stageRound = -1;
/** T-327: questions marked `wait` about a round (or version) this worker has not reached yet,
 * answered the moment it has, before it starts the next round */
let waiting: Query[] = [];
type Query = Exclude<ToDemand, { kind: "init" | "open" | "solve" | "go" | "close" | "corridorsAgain" }>;
/** T-327: the round every worker may start, by version and group (a solve marked `barrier` waits
 * for it) */
let go = { version: -1, group: 0, round: 0 };
const goAllows = (version: number, group: number, round: number) =>
  go.version > version || (go.version === version && (go.group > group || (go.group === group && go.round >= round)));
/** T-402: the solve in hand's dirty clusters by group: a solve with the barrier steps the zone in
 * view's cluster alone through its rounds first, then the rest together. `stageGroup` is the group
 * being solved; `ready` the clusters set up on the network in hand */
const groupOf = new Map<JobCluster, number>();
let stageGroup = 0;
const ready = new Set<JobCluster>();
/** the group the city's cluster is solved in (undefined: kept, or in no cluster) */
const cityGroup = (city: CityId) => { const c = jobClusters.find((c) => c.cities.includes(city)); return c ? groupOf.get(c) : undefined; };
/** a paused round boundary's resume (a `go` or a newer network) */
let wake: (() => void) | null = null;
/** the questions whose answer reads the periods' state at a round (the others read none) */
const BY_ROUND = new Set<Query["kind"]>(["station", "flowRail", "purposes", "icSel", "icStation", "transfers", "trip", "icTrip"]);
type Population = DemandApi | EverydayApi | IntercityApi;
/** T-326: the network in hand's clusters, each with its populations (cities, their everyday
 * layers, the market: the order their loads are added in), and the whole network's route nodes */
/** `pops` in the order their figures are added up; `solveOrder` the same populations in the order
 * they are set up and solved (it changes no figure) */
/** `reuse`: this worker's periods (bit q) kept from the last solve, set when it is set up (T-124) */
/** `part`: the market's part for the cluster's network (T-409), among `pops` */
/** `views`: the network each population is set on (T-756: a city's stations within its reach, the
 * market's the cluster's) and its maps to the whole network's stations and route nodes */
type JobCluster = Cluster & { pops: Population[]; solveOrder: Population[]; reuse: number; part: IntercityApi | null; views: Map<Population, PopView> };
const viewOf = (c: JobCluster, p: Population) => c.views.get(p)!;
/** T-1255: a population's boardings per period, on its own route nodes, from its last solve: the
 * free-flow pass's, and the last crowding round's before the cap. */
const boards = new WeakMap<Population, { free: (Float32Array | undefined)[]; rounds: (Float32Array | undefined)[] }>();
const boardsOf = (p: Population) => {
  let b = boards.get(p);
  if (!b) boards.set(p, (b = { free: [], rounds: [] }));
  return b;
};
/** T-756: whole network route node -> a view's, or -1 (made when first asked) */
const nodeLocals = new WeakMap<PopView, Int32Array>();
function nodeLocal(v: PopView, nodes: number): Int32Array {
  let m = nodeLocals.get(v);
  if (!m) {
    m = new Int32Array(nodes).fill(-1);
    v.nodeGlobal.forEach((g, k) => (m![g] = k));
    nodeLocals.set(v, m);
  }
  return m;
}
let jobClusters: JobCluster[] = [];
let jobNodes = 0;
/** T-326: the cluster `sig` each population holds every one of this worker's periods solved for,
 * to the last round */
const held = new Map<Population, string>();
/** T-124: per population and period, the cluster's `psig` it holds that period solved for, to the
 * last round, in a solve where every population of the cluster solved it too */
const heldQ = new Map<Population, string[]>();
const boxes = new Map<CityId, Box>();
/** T-392: the clusters of the last few networks, so a question about a shown result whose cluster
 * this worker still holds is answered at once, whatever else is being solved. `railed`: the network
 * has stations and lines (a population set on it then has subzones). */
interface Version { clusters: JobCluster[]; stations: number; railed: boolean; nodes: number; rounds: number }
const versions = new Map<number, Version>();
const KEEP_VERSIONS = 6;
/** every population of the cluster holds its network solved to the last round */
const isHeld = (c: JobCluster) => c.pops.length > 0 && c.pops.every((p) => held.get(p) === c.sig);
/** T-325, the aux worker: each solving worker's periods as a bit mask, in worker order */
let groupMasks: number[] = [];
/** T-325, the aux worker: zones open bare, apart from `cities` (a solve never covers these) */
const bare = new Map<CityId, { a: DemandApi; e: EverydayApi | null; figures: BareFigures }>();

function sendMessage(msg: FromDemand, transfer: Transferable[] = []) {
  postMessage(msg, { transfer });
}

/** Yield to the event loop so a waiting message (a newer network) is delivered. */
function yieldNow(): Promise<void> {
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => resolve();
    ch.port2.postMessage(0);
  });
}

/**
 * Open city packs and, on a solving worker, the regional market. Returns each opened city's info
 * (DemandApi.info(), with its everyday purposes folded into its trips and walk share). Any pack
 * carrying the everyday arrays gets the layer (every registered city since T-142); a pack
 * without them stays commute-only.
 */
function openPacks(packs: { id: CityId; header: string; bin: ArrayBuffer }[], market: MarketPack[] | undefined, isBare = false, rejected: OpenRejected[] = []) {
  // T-492: a pack that does not open is left out and reported; the pool carries on
  const reject = (id: CityId, isMarket: boolean, err: unknown) => rejected.push({ id, market: isMarket, error: String(err) });
  for (const pack of packs) {
    try {
      if (isBare) {
        if (!bare.has(pack.id)) bare.set(pack.id, openBare(pack));
        continue;
      }
      // (T-301: reopened before its close took effect: it stays)
      if (cities.has(pack.id)) { closeLater.ids.delete(pack.id); continue; }
      const bin = new Uint8Array(pack.bin);
      const a = new DemandApi(pack.header, bin);
      // T-399: the everyday layer shares the commutes' cells, zones, gravity and routes
      let e: EverydayApi | null = null;
      try {
        if (!aux && hasEveryday(pack.header)) e = EverydayApi.beside_with(a, pack.header, bin, attractionsIn(pack.id));
      } catch (err) {
        a.free();
        throw err;
      }
      cities.set(pack.id, a);
      if (e) everyday.set(pack.id, e);
      sendMessage({ kind: "packOpened", id: pack.id }); // T-1195
    } catch (err) {
      reject(pack.id, false, err);
    }
  }
  if (market?.length && !aux) {
    if (!marketPacks) {
      if (!marketRows) {
        for (const m of market) reject(m.id, true, "the market's rows have not arrived");
        return opened(packs, isBare, rejected);
      }
      marketPacks = new MarketPacks(marketWorld(typeof marketRows === "string" ? marketRows : new TextDecoder().decode(marketRows)));
      marketRows = null;
    }
    const have = new Set(marketPacks.places());
    const added: CityId[] = [];
    for (const m of market) {
      closeLater.markets.delete(m.id);
      if (have.has(MARKET_OF[m.id])) continue;
      try {
        // T-180: the page sends the pack's bytes; decoding here keeps it off the main thread
        marketPacks.add(typeof m.pack === "string" ? m.pack : new TextDecoder().decode(m.pack));
        added.push(m.id);
        sendMessage({ kind: "packOpened", id: m.id, market: true }); // T-1195
      } catch (err) {
        reject(m.id, true, err);
      }
    }
    if (added.length) {
      try {
        recompose();
      } catch (err) {
        // composing checks each pack against the registry's row for its zone (the zone count) and
        // names no pack: the ones just added are taken out again, the market stays as it was
        for (const id of added) { marketPacks.remove(MARKET_OF[id]); reject(id, true, err); }
        recompose();
      }
    }
  }
  return opened(packs, isBare, rejected);
}

/** Each city pack that opened, with its figures (`info`). */
function opened(packs: { id: CityId }[], isBare: boolean, rejected: OpenRejected[]) {
  const failed = new Set(rejected.filter((r) => !r.market).map((r) => r.id));
  return Object.fromEntries(packs.filter(({ id }) => !failed.has(id)).map(({ id }) => {
    const b = isBare ? bare.get(id) : undefined;
    const info = Array.from((b?.a ?? cities.get(id)!).info());
    // Auxiliary worker only serves commute allocation; compute workers report all local demand
    // (and so does the aux worker for a zone it holds bare, T-325).
    const extra = (b ? b.e : everyday.get(id))?.info();
    if (extra) {
      let outings = 0, walking = 0;
      for (let k=0;k<extra.length;k+=3) { outings += extra[k]; walking += extra[k]*extra[k+1]; }
      info[3] = (info[2]*info[3]+walking)/(info[2]+outings);
      info[2] += outings;
    }
    return [id, info];
  })) as Record<CityId, number[]>;
}

/** T-263: the world's attractions, loaded with the WASM */
let attractions: Attraction[] = [];
/** The attractions in or near a zone's box, as `EverydayApi.beside_with` takes them: x, y in the
 * pack's frame and visitors a year (the layer keeps those within reach of one of its cells). */
function attractionsIn(id: CityId): Float32Array {
  const [w, s, e, n] = widenBox(zoneBox(cityById(id)), 1000);
  const out: number[] = [];
  for (const t of attractions) {
    if (t.lng < w || t.lng > e || t.lat < s || t.lat > n) continue;
    const { x, y } = lonLatToCity(t.lng, t.lat, id);
    out.push(x, y, t.visitors);
  }
  return Float32Array.from(out);
}

const hasEveryday = (header: string) => JSON.parse(header).arrays.some((a: {name: string}) => a.name === "errand_jobs");

/**
 * T-325: a zone no station reaches, opened bare. With no station within reach every trip walks or
 * drives, so its layers are set on an empty network and solved once for every period: the same
 * figures a solve of any network gives it (no subzone, no rail), worked out the same way as
 * `run` does per period (`citySummary`, `cityParts`).
 */
function openBare(pack: { id: CityId; header: string; bin: ArrayBuffer }) {
  const bin = new Uint8Array(pack.bin);
  const a = new DemandApi(pack.header, bin);
  let e: EverydayApi | null = null;
  try {
    e = hasEveryday(pack.header) ? EverydayApi.beside_with(a, pack.header, bin, attractionsIn(pack.id)) : null;
    return solveBare(a, e);
  } catch (err) {
    e?.free();
    a.free();
    throw err;
  }
}

function solveBare(a: DemandApi, e: EverydayApi | null) {
  const f0 = new Float32Array(0), u0 = new Uint32Array(0);
  a.set_network(f0, u0, u0, f0, f0, f0);
  e?.set_network(f0, u0, u0, f0, f0, f0);
  const keys = ["work", ...EVERYDAY_KEYS.slice(0, Math.min(EVERYDAY_KEYS.length, e?.purpose_count() ?? 0))];
  const figures: BareFigures = { summary: [], parts: [], keys };
  for (let q = 0; q < PERIODS.length; q++) {
    a.solve(q);
    e?.solve(q);
    const own = Array.from(a.summary(q)), s = own.slice();
    const v = e?.summary(q);
    if (v) for (const k of [1, 2, 3, 5, 10, 11, 12]) s[k] += v[k] ?? 0;
    const p = new Array(3 * keys.length).fill(0);
    for (let k = 0; k < 3; k++) p[k] = own[1 + k] ?? 0;
    const kt = e?.kind_totals(q);
    if (kt) for (let k = 0; k < Math.min(kt.length, 3 * (keys.length - 1)); k++) p[3 + k] = kt[k];
    figures.summary.push(s);
    figures.parts.push(p);
  }
  return { a, e, figures };
}

/** T-301: closes of packs the solve in hand covers, applied once it has ended */
const closeLater = { ids: new Set<CityId>(), markets: new Set<CityId>() };

/** T-377: the market of every open market pack, in place of the last. That one is freed unless a
 * part still holds it (its zones' packs are in it; it carries on until its key goes). T-1198: freed
 * before the new one is composed, so its space (about 450 MB with the world's 2,567 places) takes the
 * new one rather than the worker's memory growing by a second; and a part composed of the same packs
 * (a selected zone's market opened, then closed) serves again, composing nothing. */
function recompose() {
  const old = intercity;
  marketZones = null;
  const key = marketPacks?.places().join(",") ?? "";
  if (old && ![...marketParts.values()].includes(old)) { intercity = null; forget([old]); }
  const same = key ? [...marketParts.values()].find((p) => composedOf.get(p) === key) : undefined;
  intercity = same ?? (key ? marketPacks!.market() : null);
  if (intercity && !same) composedOf.set(intercity, key);
}
/** T-1198: the market packs (places, in order) each composition and its parts were made of */
const composedOf = new WeakMap<IntercityApi, string>();

/** T-301: take populations out of every kept network's clusters, then free them. */
function forget(pops: Population[]) {
  const gone = new Set(pops);
  for (const cs of [jobClusters, ...[...versions.values()].map((v) => v.clusters)]) for (const c of cs) {
    if (c.part && gone.has(c.part)) c.part = null;
    c.pops = c.pops.filter((p) => !gone.has(p));
    c.solveOrder = c.solveOrder.filter((p) => !gone.has(p));
  }
  for (const p of gone) { held.delete(p); heldQ.delete(p); p.free(); }
}

function dropCity(id: CityId) {
  const a = cities.get(id);
  if (!a) return;
  const e = everyday.get(id);
  cities.delete(id);
  everyday.delete(id);
  boxes.delete(id);
  if (!bare.has(id)) squares.delete(id);
  forget(e ? [e, a] : [a]);
}

/** T-377: close zones' market packs; with none left, the market and all its parts go. */
function dropMarkets(ids: CityId[]) {
  if (!marketPacks) return;
  let changed = false;
  for (const id of ids) changed = marketPacks.remove(MARKET_OF[id]) || changed;
  if (!changed) return;
  if (marketPacks.places().length) return recompose();
  const parts = [...new Set([...(intercity ? [intercity] : []), ...marketParts.values()])];
  marketParts.clear();
  intercity = null;
  marketZones = null;
  forget(parts);
}

function closePacks(msg: Extract<ToDemand, { kind: "close" }>) {
  if (msg.bare) {
    for (const id of msg.ids) {
      const b = bare.get(id);
      if (!b) continue;
      bare.delete(id);
      b.e?.free();
      b.a.free();
      if (!cities.has(id)) squares.delete(id);
    }
  } else {
    for (const id of msg.ids) if (busy && jobCities.has(id)) closeLater.ids.add(id); else dropCity(id);
    if (msg.markets?.length) { if (busy) for (const id of msg.markets) closeLater.markets.add(id); else dropMarkets(msg.markets); }
  }
  sendMessage({ kind: "closed", memBytes: wasm_memory_bytes() });
}

function closeDeferred() {
  for (const id of closeLater.ids) dropCity(id);
  closeLater.ids.clear();
  if (closeLater.markets.size) dropMarkets([...closeLater.markets]);
  closeLater.markets.clear();
}

/** T-889: `open`s not yet taken up, each with the count of `close`s that came before it. Opens
 * that arrive together are opened together: composing the market costs about as much for one pack
 * as for forty, so a big game's packs, composed one by one, took 0.7-0.9 s each. */
const openQueue: { msg: Extract<ToDemand, { kind: "open" }>; epoch: number }[] = [];
let closeEpoch = 0;

let oneByOne = false;

function openQueued(epoch: number): Promise<void> {
  // a macrotask, so the opens already posted are delivered and join this one
  return (oneByOne ? Promise.resolve() : new Promise<void>((r) => setTimeout(r, 0))).then(() => {
    const first = openQueue[0];
    // taken up by an earlier batch, or after a close that comes before it in the chain
    if (!first || first.epoch !== epoch) return;
    const isBare = !!first.msg.bare && aux;
    let n = 1;
    while (!oneByOne && n < openQueue.length && openQueue[n].epoch === epoch && (!!openQueue[n].msg.bare && aux) === isBare) n++;
    const batch = openQueue.splice(0, n).map((q) => q.msg);
    if (!started) return;
    try {
      const t0 = performance.now();
      for (const m of batch) if (m.rows && !marketPacks) marketRows = m.rows;
      const packs = batch.flatMap((m) => m.packs), market = batch.flatMap((m) => m.intercity ?? []);
      const rejected: OpenRejected[] = [];
      const cityInfo = openPacks(packs, market, isBare, rejected);
      const openMs = performance.now() - t0;
      const warmMs = warmUp();
      const figures = isBare ? Object.fromEntries(packs.filter(({ id }) => bare.has(id)).map(({ id }) => [id, bare.get(id)!.figures])) : undefined;
      const markets = market.map((m) => m.id).filter((id) => !rejected.some((r) => r.market && r.id === id));
      sendMessage({ kind: "opened", cityInfo, markets, openMs, warmMs, memBytes: wasm_memory_bytes(), ...(figures ? { bare: figures } : {}), ...(rejected.length ? { rejected } : {}) });
    } catch (err) {
      sendMessage({ kind: "failed", error: String(err) });
    }
  });
}

/** Warm-up solves on the first city a worker opens, so V8 has optimised the kernel before the first real one. */
function warmUp(): number {
  if (warmed || !cities.size) return 0;
  warmed = true;
  return [...cities.values()].reduce((sum, a) => sum + a.warm_up(), 0);
}

onmessage = (e: MessageEvent<ToDemand>) => {
  const msg = e.data;
  if (msg.kind === "init") {
    opening = opening.then(async () => {
      try {
        const t0 = performance.now();
        [, attractions] = await Promise.all([init(), loadAttractions()]);
        const t1 = performance.now();
        aux = !!msg.aux;
        oneByOne = !!msg.oneByOne;
        mine = msg.periods;
        groupMasks = (msg.groups ?? []).map((g) => g.reduce((m, q) => m | (1 << q), 0));
        // T-173: one worker opens a small synthetic everyday layer first, so the browser has
        // optimised that code before the real cities open in every worker at once
        // (150: 22,500 cells, 900 zones; about 150 ms, mostly while the packs download)
        const everydayWarmMs = msg.warmEveryday ? everyday_warm_up(150) : 0;
        if (msg.rows && !marketPacks) marketRows = msg.rows;
        const cityInfo = openPacks(msg.packs, msg.intercity);
        const t2 = performance.now();
        const warmMs = warmUp() + everydayWarmMs;
        started = true;
        sendMessage({ kind: "ready", cityInfo, wasmMs: t1 - t0, openMs: t2 - t1, warmMs, info: cityInfo[HOME_CITY], periods: Array.from(demand_periods()), days: Array.from(day_factors()), memBytes: wasm_memory_bytes() });
      } catch (err) {
        sendMessage({ kind: "failed", error: String(err) });
      }
    });
  } else if (msg.kind === "open") {
    // T-149: packs arrive one by one, the viewed city's first; a worker busy solving opens them
    // between populations (its job keeps the cities it started with); T-889: those waiting together
    // open together
    const epoch = closeEpoch;
    openQueue.push({ msg, epoch });
    opening = opening.then(() => openQueued(epoch));
  } else if (msg.kind === "close") {
    closeEpoch++;
    // in the opening chain, so a close and a reopen of the same pack keep their order
    opening = opening.then(() => {
      if (!started) return;
      try { closePacks(msg); } catch (err) { sendMessage({ kind: "failed", error: String(err) }); }
    });
  } else if (msg.kind === "solve") {
    pending = msg;
    wake?.();
    void opening.then(() => { if (started && !busy) run(); });
  } else if (msg.kind === "go") {
    const group = msg.group ?? 0;
    if (!goAllows(msg.version, group, msg.round)) go = { version: msg.version, group, round: msg.round };
    wake?.();
  } else if (msg.kind === "corridorsAgain") {
    sentTables.clear();
  } else {
    query(msg);
  }
};

/**
 * Each local cell's 600 m square (the longer-trip market's endpoint sites are built from the same
 * cells, pipeline/intercity.py), and the commuters living and working in each square, so a
 * selection of cells can be turned into the share of every square it holds (T-118). Built once
 * per city on the first longer-trip query.
 */
interface Squares {
  key: Int32Array;
  total: Map<number, [number, number]>;
}
const SQUARE_M = 600, KEY_SPAN = 4096, KEY_OFFSET = 2048;
const squares = new Map<CityId, Squares>();
function squaresOf(id: CityId, a: DemandApi): Squares {
  let s = squares.get(id);
  if (s) return s;
  const xy = a.cell_xy(), w = a.cell_commuters(), n = xy.length / 2;
  const key = new Int32Array(n), total = new Map<number, [number, number]>();
  for (let c = 0; c < n; c++) {
    const p = cityToMarket(xy[2 * c], xy[2 * c + 1], id);
    const k = (Math.floor(p.x / SQUARE_M) + KEY_OFFSET) * KEY_SPAN + Math.floor(p.y / SQUARE_M) + KEY_OFFSET;
    key[c] = k;
    const t = total.get(k) ?? [0, 0];
    t[0] += w[c]; t[1] += w[n + c];
    total.set(k, t);
  }
  squares.set(id, (s = { key, total }));
  return s;
}
/** A selection's squares: their centres (world metres) and the share of each that is selected. */
function selectionSquares(id: CityId, a: DemandApi, end: number, cells: Uint32Array) {
  const s = squaresOf(id, a), n = s.key.length;
  const w = a.cell_commuters();
  const picked = new Map<number, number>();
  for (const c of cells) if (c < n) picked.set(s.key[c], (picked.get(s.key[c]) ?? 0) + w[end * n + c]);
  const xy = new Float32Array(2 * picked.size), share = new Float32Array(picked.size);
  let i = 0;
  for (const [k, sel] of picked) {
    const tot = s.total.get(k)![end];
    xy[2 * i] = (Math.floor(k / KEY_SPAN) - KEY_OFFSET + 0.5) * SQUARE_M;
    xy[2 * i + 1] = ((k % KEY_SPAN) - KEY_OFFSET + 0.5) * SQUARE_M;
    share[i] = tot > 0 ? Math.min(1, sel / tot) : 1;
    i++;
  }
  return { xy, share };
}
/** A zone's place in the world market (T-377: every zone has one). */
const MARKET_OF: Record<CityId, number> = Object.fromEntries(CITIES.map(c => [c.id, c.place]));
const CITY_ORDER = new Map(CITIES.map((c, i) => [c.id, i]));

/** The demand views' queries (T-078). They run between periods, never inside a solve. */
function query(msg: Query) {
  if ((msg as { wait?: boolean }).wait && msg.kind !== "cells" && BY_ROUND.has(msg.kind) && !aux) {
    const { version, round } = msg;
    // T-402: a zone in a later group of the solve in hand waits for that group's round
    const g = version === current ? cityGroup(msg.city) ?? stageGroup : stageGroup;
    const behind = version > current || (version === current && busy && (g > stageGroup || (g === stageGroup && (stageRound < round || (stageRound === round && !mine.every((q) => roundsHeld[q] === round))))));
    if (behind) return void waiting.push(msg);
  }
  // A city the network in hand was not set on (opened since, T-149) answers as if not opened.
  // (cells: a zone open bare answers too, T-325)
  // T-392: the city's cluster in the network asked about; while this worker holds it solved to the
  // last round, it answers for that network at once, whatever it is solving meanwhile
  const at = msg.kind === "cells" ? undefined : versions.get(msg.version);
  const cl = at?.clusters.find((c) => c.cities.includes(msg.city));
  const kept = !!cl && isHeld(cl);
  const a = msg.kind === "cells" ? cities.get(msg.city) ?? bare.get(msg.city)?.a : jobCities.has(msg.city) || cl ? cities.get(msg.city) : undefined;
  type WithoutCity<T> = T extends {city: CityId} ? Omit<T, "city"> : T;
  // T-285: when the question was taken up (epoch ms) and how long it took, for ?debug=1's timings
  const qStart = performance.timeOrigin + performance.now();
  const sendMessage = (reply: WithoutCity<FromDemand>, transfer: Transferable[] = []) => {
    postMessage({ ...reply, city: msg.city, qStart, qMs: performance.timeOrigin + performance.now() - qStart }, { transfer });
  };
  // T-402: the rounds this worker holds are its group's; a cluster of another group has none of them
  const atRound = (c: JobCluster | undefined) => msg.kind !== "cells" && msg.version === current && (!c || (groupOf.get(c) ?? stageGroup) === stageGroup) && mine.every(q => roundsHeld[q] === msg.round);
  const live = !!a && msg.kind !== "cells" && (atRound(cl) || (kept && msg.round === at!.rounds));
  // the network's subzones, whatever round (cellModes, flows)
  const sameNet = msg.kind !== "cells" && ((msg.version === current && (!cl || !groupOf.has(cl) || ready.has(cl))) || kept);
  // the regional market's state at the round asked about (its clusters may not be the city's):
  // every part of it for a selection, the station's own for a station
  const clusterLive = (c: JobCluster) => msg.kind !== "cells" && (atRound(c) || (isHeld(c) && msg.round === at!.rounds));
  const mcs = at?.clusters.filter((c) => c.part) ?? [];
  const marketLive = mcs.length > 0 && mcs.every(clusterLive);
  // T-260, T-464: the day asked about (0 weekday, 1 Saturday, 2 Sunday, 3 average); every answer weighs its periods by it
  const day = (msg as { day?: number }).day ?? 0;
  try {
    if (msg.kind === "cells") {
      // not open here (a focus change raced the client's wait): an empty answer, never a pool failure
      if (!a) return sendMessage({ kind: "cells", id: msg.id, xy: new Float32Array(0), h3: new BigUint64Array(0), zone: new Uint32Array(0), zoneXY: new Float32Array(1) });
      setWorldFrame(msg.frame);
      const xy = a.cell_xy(), h3 = a.cell_h3(), zone = a.cell_zone(), zoneXY = a.zone_xy();
      for (let k = 0; k < xy.length; k += 2) {
        const p = cityToWorld(xy[k], xy[k + 1], msg.city); xy[k] = p.x; xy[k + 1] = p.y;
      }
      for (let k = 1; k < zoneXY.length; k += 2) {
        const p = cityToWorld(zoneXY[k], zoneXY[k + 1], msg.city); zoneXY[k] = p.x; zoneXY[k + 1] = p.y;
      }
      sendMessage({ kind: "cells", id: msg.id, xy, h3, zone, zoneXY }, [xy.buffer, h3.buffer, zone.buffer, zoneXY.buffer]);
    } else if (msg.kind === "subSums" && day !== 0) {
      // T-260: a weekend's sums are not kept per round (nobody asks while the weekday shows); they
      // are worked out from the state now, so only while every period holds the round asked about
      const sums = live ? a!.sub_sums_day(day) : null;
      const extra = live ? everyday.get(msg.city)?.cell_modes_day(day) ?? null : null;
      sendMessage({ kind: "subSums", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, sums, everyday: extra }, [...(sums ? [sums.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "subSums") {
      // from the copy taken when the round was whole, not from the state now (see sumsByRound)
      // (or, for a cluster held to its last round, from its state, which has not moved since)
      const copy = a && msg.version === current ? sumsByRound.get(`${msg.city}:${msg.round}`) : undefined;
      const sums = copy ? copy.sums.slice() : live ? a!.sub_sums() : null;
      const extra = copy ? copy.everyday?.slice() ?? null : live ? everyday.get(msg.city)?.cell_modes() ?? null : null;
      sendMessage({ kind: "subSums", id: msg.id, version: msg.version, mask: copy ? copy.mask : live ? a!.solved_mask() : 0, sums, everyday: extra }, [...(sums ? [sums.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "cellModes") {
      // reads the network's subzones and the sums it is given, not any period's state, so it holds
      // for the version whatever round this worker is in
      const modes = a && sameNet ? a.cell_modes_day(msg.sums, day) : null;
      sendMessage({ kind: "cellModes", id: msg.id, version: msg.version, modes: modes && modes.length ? modes : null }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "station") {
      const packed = live ? stationRiders(at!, cl, a!, msg.station, day) : null;
      sendMessage({ kind: "station", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "flowRail") {
      const sums = live ? a!.flow_rail_day(msg.end, msg.cells, day) : null;
      // empty is a real answer on a network with no running line (no subzones, T-099)
      sendMessage({ kind: "flowRail", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, sums }, sums ? [sums.buffer] : []);
    } else if (msg.kind === "purposes") {
      const work = live ? a!.purpose_timing_day(msg.end, msg.cells, day) : null;
      const layer = live ? everyday.get(msg.city) : undefined;
      // T-285: packed, only the far cells the selection's samples reach (demandProtocol `everyday`)
      const extra = layer?.selection_sparse_day(msg.end, msg.cells, day) ?? null;
      const names = EVERYDAY_PURPOSES.slice(0, layer?.purpose_count() ?? 0);
      sendMessage({ kind: "purposes", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, work, everyday: extra, names }, [...(work ? [work.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "flows") {
      // reads the network's subzones and the sums it is given, not any period's state (as cellModes)
      const packed = a && sameNet ? a.flows_day(msg.end, msg.cells, msg.sums, day) : null;
      sendMessage({ kind: "flows", id: msg.id, version: msg.version, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icSel") {
      // T-336: a zone open bare in the aux worker comes with its squares (this worker does not hold
      // the zone)
      const ok = marketLive && (!!msg.sq || !!a) && !!intercity;
      const sq = !ok ? null : msg.sq ?? selectionSquares(msg.city, a!, msg.end, msg.cells);
      // T-409, T-377: the all-mode trips and the zones once, from the market of every open pack,
      // then every part's rail
      let packed: Float64Array | null = null;
      if (sq) {
        const place = MARKET_OF[msg.city];
        packed = Float64Array.from(intercity!.selection_all_day(place, sq.xy, sq.share, day, mine.reduce((m, q) => m | (1 << q), 0)));
        for (const c of mcs) {
          const v = c.part!.selection_rail_day(place, sq.xy, sq.share, day);
          for (let i = 0; i < v.length; i++) packed[i] += v[i];
        }
      }
      const mask = ok ? mcs.reduce((m, c) => m & c.part!.solved_mask(), 31) : 0;
      sendMessage({ kind: "icSel", id: msg.id, version: msg.version, mask, packed }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icStation") {
      const sc = mcs.find((c) => (c.stLocal[msg.station] ?? -1) >= 0);
      const ok = live && (!sc || clusterLive(sc)) && !!intercity;
      const packed = ok ? Float64Array.from(marketStationFlows(at!, sc, intercity!, msg.station, day)) : null;
      sendMessage({ kind: "icStation", id: msg.id, version: msg.version, mask: ok ? (sc?.part ?? intercity!).solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "purposeModes") {
      // one everyday purpose's legs (T-158); a city without that purpose answers empty, not null
      const layer = live ? everyday.get(msg.city) : undefined;
      const modes = !live ? null : layer && msg.purpose < layer.purpose_count() ? layer.cell_modes_of_day(msg.purpose, day) : new Float32Array(0);
      sendMessage({ kind: "purposeModes", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, modes }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "transfers") {
      const pairs = live ? transfersAt(at!, msg.version === current && mine.every(q => roundsHeld[q] === msg.round), msg.nodes, day) : null;
      sendMessage({ kind: "transfers", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, pairs }, pairs ? [pairs.buffer] : []);
    } else if (msg.kind === "trip") {
      // T-195: the trip as a commute, or as an everyday purpose (T-586), each period this worker
      // holds; the cells back in world metres
      // (T-979: a zone open bare is answered by the aux worker's copy, every period solved on no network)
      const b = aux ? bare.get(msg.city) : undefined;
      const api = b ? b.a : live ? a! : null;
      const answers: Float64Array[] = [], periods: number[] = [];
      if (api) {
        const o = worldToCity(msg.from[0], msg.from[1], msg.city), d = worldToCity(msg.to[0], msg.to[1], msg.city);
        const c = b ? undefined : cl;
        const k = msg.purpose ?? -1, layer = b ? b.e : everyday.get(msg.city);
        for (const q of b ? PERIODS.map((_, i) => i) : mine) {
          const v = k < 0 ? api.trip(q, o.x, o.y, d.x, d.y) : layer && k < layer.purpose_count() ? layer.trip(k, q, o.x, o.y, d.x, d.y) : new Float64Array(35);
          if (v[0] === 1) {
            const p0 = cityToWorld(v[27], v[28], msg.city), p1 = cityToWorld(v[29], v[30], msg.city);
            v[27] = p0.x; v[28] = p0.y; v[29] = p1.x; v[30] = p1.y;
            if (c) tripNodesGlobal(v, viewOf(c, api).nodeGlobal);
          }
          answers.push(v);
          periods.push(q);
        }
      }
      sendMessage({ kind: "trip", id: msg.id, version: msg.version, mask: api ? api.solved_mask() : 0, periods, answers: api ? answers : null }, answers.map((v) => v.buffer));
    } else if (msg.kind === "icTrip") {
      // T-866: each period from the market part with the most rail (each part's rail is its own
      // network's); with no part, an answer with no rail
      const ok = marketLive || (mcs.length === 0 && !!intercity && live);
      // T-885: a near place or 6 km zone instead of cells
      const near = ok ? msg.near : undefined;
      const sq = !ok || near ? null : msg.sq ?? (a ? selectionSquares(msg.city, a, msg.end, msg.cells) : null);
      const go = !!near || !!sq;
      const answers: Float64Array[] = [], periods: number[] = [];
      if (go) {
        const place = MARKET_OF[msg.city];
        for (const q of mine) {
          let best: Float64Array | null = null;
          for (const c of mcs) {
            const v = near ? c.part!.market_trip_between(q, day, near.place, near.gid, msg.end, msg.far, msg.farGid) : c.part!.market_trip(q, day, place, sq!.xy, sq!.share, msg.end, msg.far, msg.farGid);
            if (v[0] !== 1) continue;
            if (!best || v[3] + v[20] > best[3] + best[20]) { marketTripNodesGlobal(v, viewOf(c, c.part!).nodeGlobal); best = v; }
          }
          if (!best) { best = new Float64Array(22); best[0] = 1; best[14] = 2; best[13] = 1; best[6] = best[7] = -1; }
          answers.push(best);
          periods.push(q);
        }
      }
      const mask = !go ? 0 : mcs.length ? mcs.reduce((m, c) => m & c.part!.solved_mask(), 31) : mine.reduce((m, q) => m | (1 << q), 0);
      sendMessage({ kind: "icTrip", id: msg.id, version: msg.version, mask, periods, answers: go ? answers : null }, answers.map((v) => v.buffer));
    } else if (msg.kind === "bare") {
      // T-325: no network in it, so any version and round; the everyday legs per solving worker's
      // periods, as each would have sent them, so the main thread adds them up as it does those
      const b = bare.get(msg.city);
      const commute = !b ? null : msg.purpose < 0 ? b.a.cell_modes_day(new Float32Array(0), day) : new Float32Array(0);
      const e = msg.purpose === -2 ? null : b?.e;
      const parts = !e ? [] : groupMasks.map((m) => msg.purpose === -1 ? e.cell_modes_mask_day(m, day) : msg.purpose < e.purpose_count() ? e.cell_modes_of_mask_day(msg.purpose, m, day) : new Float32Array(0));
      sendMessage({ kind: "bare", id: msg.id, commute, masks: parts.length ? groupMasks : [], parts }, [...(commute ? [commute.buffer] : []), ...parts.map((p) => p.buffer)]);
    } else if (msg.kind === "bareFlows") {
      // T-336: a bare zone's selection with no rail: the commuters' far end as `flows` gives it
      // (no subzones, so no rail sums), and per solving worker's periods what its `purposes` would
      const b = bare.get(msg.city);
      const packed = b ? b.a.flows_day(msg.end, msg.cells, new Float32Array(0), day) : null;
      const work = b ? groupMasks.map((m) => b.a.purpose_timing_mask_day(msg.end, msg.cells, m, day)) : [];
      const e = b?.e;
      const extra = e ? groupMasks.map((m) => e.selection_sparse_mask_day(msg.end, msg.cells, m, day)) : [];
      const names = EVERYDAY_PURPOSES.slice(0, e?.purpose_count() ?? 0);
      const out = packed && packed.length ? packed : null;
      sendMessage({ kind: "bareFlows", id: msg.id, packed: out, masks: b ? groupMasks : [], work, everyday: extra, names }, [...(out ? [out.buffer] : []), ...work.map((v) => v.buffer), ...extra.map((v) => v.buffer)]);
    } else if (msg.kind === "bareSquares") {
      // T-336: the solving workers hold the market but not the zone: its squares come from here
      const b = bare.get(msg.city);
      const sq = b ? selectionSquares(msg.city, b.a, msg.end, msg.cells) : null;
      sendMessage({ kind: "bareSquares", id: msg.id, xy: sq?.xy ?? null, share: sq?.share ?? null });
    }
  } catch (err) {
    // a failed query answers with nothing; the pool itself carries on
    console.error("demand query:", err);
    if (msg.kind === "subSums" || msg.kind === "station") sendMessage({ kind: msg.kind, id: msg.id, version: msg.version, mask: 0, ...(msg.kind === "subSums" ? { sums: null, everyday: null } : { packed: null }) } as FromDemand);
    else if (msg.kind === "purposes") sendMessage({kind: "purposes", id: msg.id, version: msg.version, mask: 0, work: null, everyday: null, names: []});
    else if (msg.kind === "cellModes") sendMessage({ kind: "cellModes", id: msg.id, version: msg.version, modes: null });
    else if (msg.kind === "flowRail") sendMessage({ kind: "flowRail", id: msg.id, version: msg.version, mask: 0, sums: null });
    else if (msg.kind === "flows") sendMessage({ kind: "flows", id: msg.id, version: msg.version, packed: null });
    else if (msg.kind === "icSel" || msg.kind === "icStation") sendMessage({ kind: msg.kind, id: msg.id, version: msg.version, mask: 0, packed: null });
    else if (msg.kind === "purposeModes") sendMessage({ kind: "purposeModes", id: msg.id, version: msg.version, mask: 0, modes: null });
    else if (msg.kind === "transfers") sendMessage({ kind: "transfers", id: msg.id, version: msg.version, mask: 0, pairs: null });
    else if (msg.kind === "trip") sendMessage({ kind: "trip", id: msg.id, version: msg.version, mask: 0, periods: [], answers: null });
    else if (msg.kind === "icTrip") sendMessage({ kind: "icTrip", id: msg.id, version: msg.version, mask: 0, periods: [], answers: null });
    else if (msg.kind === "bare") sendMessage({ kind: "bare", id: msg.id, commute: null, masks: [], parts: [] });
    else if (msg.kind === "bareFlows") sendMessage({ kind: "bareFlows", id: msg.id, packed: null, masks: [], work: [], everyday: [], names: [] });
    else if (msg.kind === "bareSquares") sendMessage({ kind: "bareSquares", id: msg.id, xy: null, share: null });
  }
}

/** T-327: ask the waiting questions again; those still ahead of this worker wait on. */
function answerWaiting() {
  const w = waiting;
  waiting = [];
  for (const m of w) query(m);
}

/**
 * Transfers with either end at one of `nodes` (T-176): every population of the job, every period
 * this worker holds, each (arriving, leaving) route node pair once. Asked for one station complex
 * on selection, so the answer is a few thousand numbers, never the whole network's.
 */
function transfersAt(at: Version, allLive: boolean, nodes: Uint32Array, day: number): Float32Array {
  const sum = new Map<number, number>(), jobNodes = at.nodes;
  for (const c of at.clusters) {
    // T-392: a cluster mid-solve has no figures for this network yet (T-402: nor one of a later group)
    if (!isHeld(c) && !(allLive && (groupOf.get(c) ?? stageGroup) === stageGroup)) continue;
    for (const p of c.pops) {
      const view = viewOf(c, p), local = nodeLocal(view, jobNodes);
      const sel = new Uint8Array(view.nodeGlobal.length);
      let any = false;
      for (const r of nodes) {
        const l = r < jobNodes ? local[r] : -1;
        if (l >= 0) (sel[l] = 1), (any = true);
      }
      if (!any) continue;
      const v = p.transfers_day(sel, day);
      for (let k = 0; k + 2 < v.length; k += 3) {
        const key = view.nodeGlobal[v[k]] * jobNodes + view.nodeGlobal[v[k + 1]];
        sum.set(key, (sum.get(key) ?? 0) + v[k + 2]);
      }
    }
  }
  const out = new Float32Array(3 * sum.size);
  let o = 0;
  for (const [key, riders] of sum) {
    out[o++] = Math.floor(key / jobNodes); out[o++] = key % jobNodes; out[o++] = riders;
  }
  return out;
}

/**
 * T-326: a station's riders from one city, on its own network (T-756). A station outside it
 * gets the answer a population set on the whole network gives for a station it cannot reach: no
 * riders, every zone listed (`DemandApi.station_riders_day`).
 */
function stationRiders(at: Version, c: JobCluster | undefined, a: DemandApi, station: number, day: number): Float32Array {
  const view = c?.views.get(a);
  const ls = view && view.net.lineN.length ? view.stLocal[station] ?? -1 : -1;
  if (ls >= 0) return a.station_riders_day(ls, day);
  return at.railed && station < at.stations ? new Float32Array(4 + 2 * a.info()[1]) : new Float32Array(0);
}
/** The same for the market's longer trips (`IntercityApi.station_flows_day`), from the part of
 * the station's cluster `c` (T-409); `m` answers for a station in no market cluster. */
function marketStationFlows(at: Version, c: JobCluster | undefined, m: IntercityApi, station: number, day: number): Float64Array {
  const ls = c ? c.stLocal[station] ?? -1 : -1;
  if (ls >= 0) return c!.part!.station_flows_day(ls, day);
  return station < at.stations ? m.no_station_flows() : new Float64Array(0);
}

/**
 * T-1008: the market's corridor table (game/demand.ts `CorridorTable`): the pairs with longer trips
 * or commutes, which depend on neither the period nor the network (most of the market's place
 * pairs have none), with a key of all of it. `row` maps a market corridor to its row, -1 none.
 * Made once per market part; the main thread is sent each key once (`sentTables`), since its tens
 * of MB in every period's result cost it a frame or more each (T-790). T-1204: the parts and
 * compositions of one world share their corridors (`corridors_id`), so the table is made once for
 * them all (it was tens of MB a part, and as many again to make).
 */
interface MadeTable { table: CorridorTable; modes: Float32Array; row: Int32Array }
let madeTable: { id: number; made: MadeTable } | null = null;
const sentTables = new Set<string>();
function corridorTable(p: IntercityApi): MadeTable {
  const id = p.corridors_id();
  if (madeTable?.id === id) return madeTable.made;
  const pairs = p.corridor_pairs(), trips = p.corridor_trips(), base = p.commute_base(), modes = p.corridor_modes();
  const N = trips.length, hasK = base.length === N, hasM = modes.length === 2 * N;
  const row = new Int32Array(N).fill(-1);
  let n = 0;
  for (let i = 0; i < N; i++) if (trips[i] > 0 || (hasK && base[i] > 0)) row[i] = n++;
  const t: CorridorTable = { key: "", n, from: new Uint32Array(n), to: new Uint32Array(n), longer: new Float64Array(n), commutes: new Float64Array(n), air: new Float32Array(n), car: new Float32Array(n) };
  for (let i = 0; i < N; i++) {
    const j = row[i];
    if (j < 0) continue;
    t.from[j] = pairs[2 * i]; t.to[j] = pairs[2 * i + 1]; t.longer[j] = trips[i];
    if (hasK) t.commutes[j] = base[i];
    // T-378: before any shares are known, all by car
    if (hasM) { t.air[j] = modes[2 * i]; t.car[j] = modes[2 * i + 1]; } else t.car[j] = 1;
  }
  // FNV-1a over the bits of every column
  let h = 0x811c9dc5;
  for (const a of [t.from, t.to, t.longer, t.commutes, t.air, t.car]) {
    const w = new Uint32Array(a.buffer);
    for (let i = 0; i < w.length; i++) h = Math.imul(h ^ w[i], 0x01000193);
  }
  t.key = `${n}:${(h >>> 0).toString(36)}`;
  const made = { table: t, modes, row };
  madeTable = { id, made };
  return made;
}

/** T-1008: period `q`'s corridor figures over the market's parts: the table, the period's shares
 * (the first part's that solved it; the trips are the same in every part) and the pairs with rail,
 * each part's own network's added up, by table row, ascending. */
function marketCorridors(parts: IntercityApi[], q: number) {
  if (!parts.length) return null;
  const made = corridorTable(parts[0]);
  let shares: number[] = [];
  const rail = new Map<number, [number, number]>();
  for (const p of parts) {
    const s = p.corridor_shares(q);
    if (!s.length) continue;
    if (!shares.length) shares = Array.from(s);
    const v = p.corridor_rail(q);
    for (let i = 0; i < v.length; i += 3) {
      const j = made.row[v[i]];
      if (j < 0) continue;
      const r = rail.get(j);
      if (r) { r[0] += v[i + 1]; r[1] += v[i + 2]; } else rail.set(j, [v[i + 1], v[i + 2]]);
    }
  }
  const rows = [...rail.keys()].sort((a, b) => a - b);
  const out = new Float64Array(3 * rows.length);
  rows.forEach((j, k) => { const [r, c] = rail.get(j)!; out[3 * k] = j; out[3 * k + 1] = r; out[3 * k + 2] = c; });
  const t = made.table, first = !sentTables.has(t.key);
  sentTables.add(t.key);
  // copies, transferred: the worker keeps its own for `corridorsAgain`
  const table: CorridorTable | undefined = first ? { ...t, from: t.from.slice(), to: t.to.slice(), longer: t.longer.slice(), commutes: t.commutes.slice(), air: t.air.slice(), car: t.car.slice() } : undefined;
  return { corridorKey: t.key, corridorTable: table, corridorModes: first ? made.modes.slice() : undefined, corridorShares: shares.length >= 2 ? [shares[0], shares[1]] as [number, number] : undefined, corridorRail: out };
}

/**
 * T-409: the market's figures in period q over its parts (`jobClusters` order). Each part's rail
 * is its own network's; the trips, pairs and corridor trips need no network and are the same in
 * every part, so they are taken once. One part gives its own figures as they are.
 */
function marketFigures(parts: IntercityApi[], q: number) {
  const of = (read: (p: IntercityApi) => Float64Array, rail: (i: number) => boolean) => {
    let out: number[] = [];
    for (const p of parts) {
      const v = read(p);
      if (!v.length) continue;
      if (!out.length) out = Array.from(v);
      else for (let i = 0; i < v.length; i++) if (rail(i)) out[i] += v[i];
    }
    return out;
  };
  // round, trips, rail, 0, ms, pairs, segments over seats, then the worst loads: the largest part's;
  // then the fare sensitivity (T-1300)
  const summary = of((p) => p.summary(q), (i) => i === 2 || i === 4 || i === 6 || i >= 10);
  if (parts.length > 1 && summary.length) for (const i of [7, 8, 9]) summary[i] = Math.max(...parts.map((p) => p.summary(q)[i] ?? 0));
  // the longer trips, the commutes, the everyday pairs (T-1175), then (T-1229) the pairs by purpose,
  // four each: only the rail is a part's own
  const ps = of((p) => p.part_summary(q), (i) => (i < 6 ? i % 2 === 1 : i >= 8 && (i - 8) % 4 === 1));
  // T-1175: the everyday pairs' walkers are every pair's with no rail (the same in every part) less
  // those each part's rail takes, in all and for each purpose
  if (parts.length > 1 && ps.length >= 7) {
    const each = parts.map((p) => p.part_summary(q)).filter((v) => v.length >= 7);
    for (const [at, base] of [[6, 7], ...Array.from({ length: Math.max(0, (ps.length - 8) >> 2) }, (_, k) => [8 + 4 * k + 2, 8 + 4 * k + 3])]) {
      const b = each[0][base] ?? each[0][at];
      ps[at] = Math.max(0, b - each.reduce((s, v) => s + (b - v[at]), 0));
    }
    if (summary.length) summary[3] = ps[6];
  }
  return {
    summary,
    parts: ps,
    ...marketCorridors(parts, q),
    // T-378: the same in every part
    elsewhere: parts.length ? Array.from(parts[0].elsewhere(q)) : [],
  };
}
/** A `DemandApi.trip` answer's route nodes (trip.ts `PAIR_LEN`, `LEG_LEN`) from its cluster's
 * network to the whole network's, the ones the client reads. */
function tripNodesGlobal(v: Float64Array, nodeGlobal: Uint32Array) {
  const { HEAD, PAIR_LEN, LEG_LEN } = TRIP_LAYOUT;
  const g = (i: number) => { if (v[i] >= 0) v[i] = nodeGlobal[v[i]]; };
  const np = v[24], nr = v[25];
  for (let k = 0; k < np; k++) { g(HEAD + k * PAIR_LEN + 2); g(HEAD + k * PAIR_LEN + 3); }
  let o = HEAD + np * PAIR_LEN;
  for (let k = 0; k < nr; k++) {
    const n = v[o + 2];
    o += 3;
    for (let i = 0; i < n; i++, o += LEG_LEN) { g(o + 1); g(o + 2); }
  }
}

/** The same for an `IntercityApi.market_trip` answer (T-866: head 22 since T-1229, pairs of 7, legs of 12). */
function marketTripNodesGlobal(v: Float64Array, nodeGlobal: Uint32Array) {
  const HEAD = 22, PAIR_LEN = 7, { LEG_LEN } = TRIP_LAYOUT;
  const g = (i: number) => { if (v[i] >= 0) v[i] = nodeGlobal[v[i]]; };
  const np = v[11], nr = v[12];
  for (let k = 0; k < np; k++) { g(HEAD + k * PAIR_LEN); g(HEAD + k * PAIR_LEN + 1); }
  let o = HEAD + np * PAIR_LEN;
  for (let k = 0; k < nr; k++) {
    const n = v[o + 2];
    o += 3;
    for (let i = 0; i < n; i++, o += LEG_LEN) { g(o + 1); g(o + 2); }
  }
}

/**
 * A period's `seg`, `board` and `alight` split by trip purpose (T-158, demandProtocol `parts`):
 * work is every city's commutes, then the everyday layer's purposes, then the regional market's
 * longer trips. They add up to the result's totals; crowding is never split. Each cluster's route
 * nodes go to their place in the whole network's (T-326).
 */
function purposeParts(q: number, nr: number, clusters: typeof jobClusters, regional: boolean) {
  const days = clusters.flatMap((c) => c.pops.filter((p): p is EverydayApi => p instanceof EverydayApi));
  const kinds = Math.min(EVERYDAY_KEYS.length, Math.max(0, ...days.map((e) => e.purpose_count())));
  const keys: string[] = ["work", ...EVERYDAY_KEYS.slice(0, kinds)];
  if (regional) keys.push("longer");
  const seg = new Float32Array(keys.length * nr), board = new Float32Array(keys.length * nr), alight = new Float32Array(keys.length * nr);
  for (const c of clusters) {
    // T-756: each population's route nodes go through its own view
    let map = c.nodeGlobal, m = map.length;
    const to = (p: Population) => { map = viewOf(c, p).nodeGlobal; m = map.length; };
    const add = (out: Float32Array, at: number, v: ArrayLike<number>, from = 0) => {
      if (v.length < from + m) return; // nothing solved for this population (no network)
      for (let r = 0; r < m; r++) out[at * nr + map[r]] += v[from + r];
    };
    for (const a of c.pops) if (a instanceof DemandApi) { to(a); add(seg, 0, a.seg(q)); add(board, 0, a.board(q)); add(alight, 0, a.alight(q)); }
    for (const e of c.pops) {
      if (!(e instanceof EverydayApi)) continue;
      to(e);
      const v = e.part_loads(q), k1 = Math.min(kinds, e.purpose_count());
      if (v.length !== 3 * e.purpose_count() * m) continue;
      for (let k = 0; k < k1; k++) { add(seg, 1 + k, v, 3 * k * m); add(board, 1 + k, v, (3 * k + 1) * m); add(alight, 1 + k, v, (3 * k + 2) * m); }
    }
    if (regional && c.part) {
      const at = keys.length - 1, part = c.part;
      to(part);
      add(seg, at, part.seg(q)); add(board, at, part.board(q)); add(alight, at, part.alight(q));
      // T-276: the regional layer's commutes between zones are work: moved from longer trips to it
      const v = part.commute_loads(q);
      const move = (v: Float32Array, from: number, to: number) => {
        for (let r = 0; r < m; r++) {
          const g = map[r];
          seg[to * nr + g] += v[from + r]; seg[at * nr + g] -= v[from + r];
          board[to * nr + g] += v[from + m + r]; board[at * nr + g] -= v[from + m + r];
          alight[to * nr + g] += v[from + 2 * m + r]; alight[at * nr + g] -= v[from + 2 * m + r];
        }
      };
      if (v.length === 3 * m) move(v, 0, 0);
      // T-1175: its everyday outings across a zone line are their purposes', a block each
      const e = part.everyday_loads(q);
      if (e.length && e.length % (3 * m) === 0) for (let k = 0; k < Math.min(kinds, e.length / (3 * m)); k++) move(e, 3 * k * m, 1 + k);
    }
  }
  return { keys, seg, board, alight };
}

/**
 * T-409: a market part for every market cluster's key: the part that had the key, else one of the
 * newest composition whose key is gone (set up afresh), the composed market itself first, else a
 * new one beside it. Parts left over are freed (never the composed market itself).
 */
function assignParts(regional: IntercityApi, split: Cluster[]) {
  const keys = [...new Set(split.filter((c) => c.market).map((c) => c.marketKey))];
  const spare = [...marketParts].filter(([k]) => !keys.includes(k)).map(([k, p]) => (marketParts.delete(k), p));
  const used = new Set(marketParts.values());
  const usable = spare.filter((p) => p !== regional && regional.same_market(p));
  if (!used.has(regional)) usable.unshift(regional);
  for (const k of keys) {
    if (marketParts.has(k)) continue;
    let p = usable.shift();
    if (!p) { p = regional.beside(); composedOf.set(p, composedOf.get(regional) ?? ""); }
    marketParts.set(k, p);
    used.add(p);
  }
  for (const p of new Set(spare)) {
    if (used.has(p) || p === regional) continue;
    held.delete(p);
    heldQ.delete(p);
    p.free();
  }
}

async function run() {
  busy = true;
  try {
    while (pending && started) {
      closeDeferred();
      const job = pending;
      pending = null;
      const n = job.net;
      setWorldFrame(n.frame);
      let setupMs = 0;
      // the cities open now: a pack opened during this job waits for the next one (T-149)
      // in the registry's order, not the order the packs happened to open in: the figures are added
      // up in it, so the same network gives the same figures to the bit however the packs arrived
      const byRegistry = <T,>(list: [CityId, T][]) => list.sort((x, y) => CITY_ORDER.get(x[0])! - CITY_ORDER.get(y[0])!);
      const jobLocal = byRegistry([...cities]);
      const jobEveryday = byRegistry([...everyday].filter(([id]) => cities.has(id)));
      jobCities = new Set(jobLocal.map(([id]) => id));
      const regional = intercity;
      // T-326: each cluster on its own network; one this worker has solved on the same network
      // keeps its populations' results, untouched
      for (const [id, a] of jobLocal) if (!boxes.has(id)) boxes.set(id, cellBox(a.cell_xy()));
      if (regional && !marketZones) marketZones = regional.zone_boxes();
      const split = splitClusters(n, jobLocal.map(([id]) => ({ id, box: boxes.get(id) ?? null })), regional ? marketZones : null, job.rounds, job.whole, job.oneMarket);
      if (regional) assignParts(regional, split);
      const partOf = (c: Cluster) => (c.market && regional ? [marketParts.get(c.marketKey)!] : []);
      jobClusters = split.map((c) => ({ ...c, part: partOf(c)[0] ?? null, pops: [
        ...c.cities.map((id) => cities.get(id)!),
        ...jobEveryday.filter(([id]) => c.cities.includes(id)).map(([, e]) => e),
        ...partOf(c),
      ], solveOrder: [
        // the order the packs opened in (the viewed zone's first); the order changes no figure. Set
        // up in registry order instead, the real save's first round-0 solve on each worker is ~1 s
        // slower in WASM (T-405, notes/T-405.md; the skim cache no longer depends on the order)
        ...[...cities.keys()].filter((id) => c.cities.includes(id)).map((id) => cities.get(id)!),
        ...[...everyday.keys()].filter((id) => c.cities.includes(id) && jobCities.has(id)).map((id) => everyday.get(id)!),
        ...partOf(c),
      ], reuse: 0, views: new Map<Population, PopView>() }));
      // T-756: a city's layers solve on the stations within its reach (`?perfOff=reach`: on its
      // whole cluster); the market's longer trips on the whole cluster
      for (const c of jobClusters) {
        const whole = clusterView(c);
        for (const id of c.cities) {
          const v = job.wide || job.whole ? whole : reachView(n, c, cityReach(n, id, boxes.get(id) ?? null));
          c.views.set(cities.get(id)!, v);
          const e = everyday.get(id);
          if (e && c.pops.includes(e)) c.views.set(e, v);
        }
        if (c.part) c.views.set(c.part, whole);
      }
      const dirty = jobClusters.filter((c) => c.pops.some((p) => held.get(p) !== c.sig));
      // T-393: the cluster of the city in view first
      if (job.first) {
        const i = dirty.findIndex((c) => c.cities.includes(job.first!));
        if (i > 0) dirty.unshift(...dirty.splice(i, 1));
      }
      // T-392: after an edit (no barrier) the clusters are solved one after another, each to its last
      // round, and each but the last is sent as a cluster result the moment it is done; then the
      // solve's results. With the barrier (the shown estimates step through the rounds) they all go
      // round by round together, in groups (T-402): the city in view's cluster alone first, sent as
      // cluster results round by round, then the rest, whose results are the solve's.
      const early = job.barrier ? [] : dirty;
      const ahead = job.barrier && !job.whole && dirty.length > 1 && !!job.first && dirty[0].cities.includes(job.first);
      const groups = !job.barrier ? [] : ahead ? [[dirty[0]], dirty.slice(1)] : [dirty];
      groupOf.clear();
      ready.clear();
      for (const c of early) groupOf.set(c, 0);
      groups.forEach((cs, g) => { for (const c of cs) groupOf.set(c, g); });
      stageGroup = 0;
      let periodsKept = 0, periodsSolved = 0;
      const setUp =(c: JobCluster) => {
        for (const p of c.solveOrder) {
          const m = viewOf(c, p).net;
          held.delete(p);
          p.set_complexes(m.complex);
          p.set_upgrades(m.ups ?? new Uint8Array(0));
          p.set_passages(m.passages ?? new Uint32Array(0));
          p.set_span(m.span ?? new Float32Array(0));
          // T-1277: riders weigh fares by their country's income and pay the curve times its scale;
          // the market takes the curve as set, at the baseline country's weight, and (T-1286) each
          // trip's at its origin's scale over its income's
          let scale = 1;
          if (p instanceof IntercityApi) {
            p.set_fare_weight(fareWeight(FARE_BASELINE));
            p.set_fare_factors(JSON.stringify(fareFactors(m.fare.scales)));
            p.set_sampled(!job.exactMarket);
            // T-377: the market takes stations as longitude and latitude, each zone in its own frame
            const ll = new Float64Array(m.stXY.length);
            for (let k = 0; k < ll.length; k += 2) [ll[k], ll[k + 1]] = worldToLonLat(m.stXY[k], m.stXY[k + 1]);
            setupMs += p.set_network_ll(ll, m.lineN, m.stops, m.times, m.tph, m.cars);
          } else {
            if (p instanceof DemandApi) p.set_chunked(!job.exactMarket);
            const city = c.cities.find((id) => cities.get(id) === p || everyday.get(id) === p)!;
            const country = cityById(city).country;
            p.set_fare_weight(fareWeight(country));
            scale = fareScale(country, m.fare.scales);
            const xy = m.stXY.slice();
            for (let k = 0; k < xy.length; k += 2) {
              const w = worldToCity(xy[k], xy[k + 1], city);
              xy[k] = w.x; xy[k + 1] = w.y;
            }
            setupMs += p.set_network(xy, m.lineN, m.stops, m.times, m.tph, m.cars);
          }
          p.set_fares(m.fare.base * scale, m.fare.perKm * scale, m.hopKm);
          p.set_fare_curve(scaledKnots(m.fare, scale));
        }
        // T-124: a period is kept only if every population kept it (its inputs are unchanged: the
        // Rust side) and each holds it from a whole solve of this cluster's period (`psig`): the
        // crowding is the whole cluster's
        let reuse = job.fresh ? 0 : mine.reduce((r, q) => r | (1 << q), 0);
        for (const p of c.pops) {
          reuse &= p.solved_mask();
          const h = heldQ.get(p);
          for (const q of mine) if (h?.[q] !== c.psig[q]) reuse &= ~(1 << q);
        }
        for (const p of c.pops) {
          const h = heldQ.get(p) ?? [];
          for (const q of mine) if (!(reuse & (1 << q))) h[q] = "";
          heldQ.set(p, h);
        }
        c.reuse = reuse;
        ready.add(c);
        for (const q of mine) if (reuse & (1 << q)) periodsKept++; else periodsSolved++;
      };
      const solved =(c: JobCluster) => {
        for (const p of c.pops) {
          held.set(p, c.sig);
          const h = heldQ.get(p)!;
          for (const q of mine) h[q] = c.psig[q];
        }
      };
      jobNodes = 2 * n.stops.length;
      versions.set(job.version, { clusters: jobClusters, stations: n.stXY.length / 2, railed: n.stXY.length > 0 && n.lineN.length > 0, nodes: jobNodes, rounds: job.rounds });
      for (const v of versions.keys()) if (versions.size > KEEP_VERSIONS && v !== job.version) versions.delete(v);
      sendMessage({ kind: "plan", version: job.version, solving: dirty.map((c) => c.cities), kept: jobClusters.filter((c) => !dirty.includes(c)).flatMap((c) => c.cities), ...(ahead ? { ahead: dirty[0].cities } : {}) });
      current = job.version;
      roundsHeld.fill(-1);
      sumsByRound.clear();
      stageRound = -1;
      answerWaiting(); // an older version's get nulls; this one's wait on
      let stale = false;
      // the clusters' loads, all their populations', per route node of the whole network
      const sumAll = (cs: JobCluster[], read: (a: Population) => Float32Array, out = new Float32Array(jobNodes), at = 0): Float32Array => {
        for (const c of cs) {
          for (const a of c.pops) { const map = viewOf(c, a).nodeGlobal, v = read(a); for (let k = 0; k < map.length; k++) out[at + map[k]] += v[k] ?? 0; }
        }
        return out;
      };
      // One round of period q for the cluster. Periods are independent. Every population of a
      // cluster advances from one frozen load vector of the whole cluster, never from whichever
      // population happened to run first. False if a newer network came meanwhile.
      const step = async (c: JobCluster, round: number, q: number): Promise<boolean> => {
        // the cluster's segment loads, boardings and alightings per route node: the crowding and
        // the waits for full trains (T-188) the round is solved with
        const N = jobNodes;
        const loads = () => {
          const t = new Float32Array(3 * N);
          sumAll([c], a => a.seg(q), t, 0); sumAll([c], a => a.board(q), t, N); sumAll([c], a => a.alight(q), t, 2 * N);
          return t;
        };
        // the cluster's loads on a population's own route nodes; at a pad, the riders aboard as
        // its boardings (`PopView.pads`)
        const ownOf = (a: Population, total: Float32Array) => {
          const { nodeGlobal: map, pads } = viewOf(c, a), nr = map.length, own = new Float32Array(3 * nr);
          for (let k = 0; k < nr; k++) { const g = map[k]; own[k] = total[g]; own[nr + k] = total[N + g]; own[2 * nr + k] = total[2 * N + g]; }
          for (const k of pads) { own[nr + k] = own[k]; own[2 * nr + k] = 0; }
          return own;
        };
        const total = round > 0 ? loads() : null;
        for (const a of c.solveOrder) {
          if (total === null) a.solve(q);
          else if (!a.crowd_with_external(q, ownOf(a, total))) throw new Error("shared crowding layout disagrees");
          // a population's pass is up to a second of one call: let a waiting query or a newer
          // network in between
          await yieldNow();
          if (pending) return false;
        }
        // T-1255: the free-flow pass's boardings, the riders who would come if the trains had room
        if (total === null) for (const a of c.pops) { const b = boardsOf(a); b.free[q] = a.board(q); b.rounds[q] = undefined; }
        if (round === job.rounds && round > 0) {
          // T-1070, T-1255: the final loads, every population's, held to the riders the trains take
          // within the wait for a train with room, with everyone who comes to the platforms
          const final = loads();
          for (const a of c.pops) boardsOf(a).rounds[q] = a.board(q);
          const come = comeOf([c], q);
          for (const a of c.solveOrder) {
            const own = ownOf(a, final), { nodeGlobal: map, pads } = viewOf(c, a), nr = map.length, four = new Float32Array(4 * nr);
            four.set(own);
            for (let k = 0; k < nr; k++) four[3 * nr + k] = Math.max(come[map[k]], own[nr + k]);
            for (const k of pads) four[3 * nr + k] = own[nr + k];
            if (!a.cap_with_external(q, four)) throw new Error("shared capacity layout disagrees");
          }
        }
        return true;
      };
      // T-1255: per route node of the whole network, the riders coming to the platforms of clusters
      // `cs` in period q: the free-flow pass's boardings, at least the rounds' (before the cap)
      const comeOf = (cs: JobCluster[], q: number): Float32Array => {
        const free = sumAll(cs, (a) => boardsOf(a).free[q] ?? a.board(q));
        const rounds = sumAll(cs, (a) => boardsOf(a).rounds[q] ?? a.board(q));
        for (let k = 0; k < free.length; k++) free[k] = Math.max(free[k], rounds[k]);
        return free;
      };
      // period q's figures over the clusters `cs` (all of them for a result), as the state stands
      const figures = (q: number, round: number, ms: number, cs: JobCluster[]) => {
        // in the job's order: the summary adds the cities up in it
        const ids = jobLocal.map(([id]) => id).filter((id) => cs.some((c) => c.cities.includes(id)));
        // the market's figures are all its parts' as they stand, whichever clusters are asked for
        const market = regional && cs.some((c) => c.market) ? marketFigures(jobClusters.flatMap((c) => c.part ? [c.part] : []), q) : null;
        const seg = sumAll(cs, a => a.seg(q)), board = sumAll(cs, a => a.board(q)), alight = sumAll(cs, a => a.alight(q)), loadOfCrush = sumAll(cs, a => a.load_of_crush(q)), fareX = sumAll(cs, a => a.fare_extra(q));
        // the account's riders coming: everyone who would ride where the queue turned riders away
        // (their trains carry no more), the riders who ride elsewhere, so the account's boardings
        // stay demand's
        const arrive = comeOf(cs, q), rounds = sumAll(cs, (a) => boardsOf(a).rounds[q] ?? a.board(q));
        for (let k = 0; k < arrive.length; k++) arrive[k] = board[k] < 0.999 * rounds[k] ? Math.max(arrive[k], board[k]) : board[k];
        const citySummary = Object.fromEntries(ids.map((id) => [id, Array.from(cities.get(id)!.summary(q))])) as Record<CityId, number[]>;
        for (const [id, a] of jobEveryday) if (citySummary[id]) { const v = a.summary(q); for (const k of [1, 2, 3, 5, 10, 11, 12]) citySummary[id][k] += v[k] ?? 0; }
        const intercitySummary = market ? market.summary : [];
        const summary = [round, 0, 0, 0, ms, 0, 0, 0, 0, 0];
        for (const v of [...Object.values(citySummary), intercitySummary]) for (const k of [1, 2, 3, 5]) summary[k] += v[k] ?? 0;
        // T-1300: rail trips per curve dollar dearer on every fare, local and the market's (each
        // population's per perceived minute, summary slot 10, times its minutes a curve dollar), and
        // how far the trips' sensitivity reaches as against their riders: the sensitivity-weighted
        // mean km over the rider-weighted one (slots 11 and 12)
        const perDollar = (id: CityId) => { const c = cityById(id).country; return fareWeight(c) * fareScale(c, n.fare.scales); };
        const reach = (d: number, dKm: number, rail: number, km: number) => (d < 0 && km > 0 && rail > 0 ? (dKm / d) / (km / rail) : 1);
        const ls = ids.reduce((a, id) => {
          const v = citySummary[id], w = perDollar(id);
          return [a[0] + (v[10] ?? 0) * w, a[1] + (v[11] ?? 0) * w, a[2] + (v[2] ?? 0), a[3] + (v[12] ?? 0)];
        }, [0, 0, 0, 0]);
        const im = intercitySummary;
        const fareSense: [number, number, number, number] = [ls[0], (im[10] ?? 0) * fareWeight(FARE_BASELINE), reach(ls[0], ls[1], ls[2], ls[3]), reach(im[10] ?? 0, im[11] ?? 0, im[2] ?? 0, im[12] ?? 0)];
        const parts = purposeParts(q, jobNodes, cs, !!market);
        // T-260: each city's trips, rail and walked trips by purpose, `parts.keys` order less the
        // market, so the main thread can work out a weekend's mode split
        const local = parts.keys.length - (market ? 1 : 0);
        const cityParts = Object.fromEntries(ids.map((id) => {
          const v = new Array(3 * local).fill(0), s = cities.get(id)!.summary(q);
          for (let k = 0; k < 3; k++) v[k] = s[1 + k] ?? 0;
          const e = everyday.get(id)?.kind_totals(q);
          if (e) for (let k = 0; k < Math.min(e.length, 3 * (local - 1)); k++) v[3 + k] = e[k];
          return [id, v];
        })) as Record<CityId, number[]>;
        const msg = { version: job.version, period: q, summary, fareSense, citySummary, cityParts, intercitySummary, intercityParts: market ? market.parts : [], corridorKey: market?.corridorKey, corridorTable: market?.corridorTable, corridorShares: market?.corridorShares, corridorRail: market?.corridorRail, corridorModes: market?.corridorModes, elsewhere: market?.elsewhere, seg, board, alight, arrive, loadOfCrush, fareX, parts, ms, setupMs, memBytes: wasm_memory_bytes() };
        const t = market?.corridorTable;
        const big = [market?.corridorRail?.buffer, market?.corridorModes?.buffer, ...(t ? [t.from, t.to, t.longer, t.commutes, t.air, t.car].map((a) => a.buffer) : [])].filter((b): b is ArrayBuffer => !!b);
        return { msg, transfer: [seg.buffer, board.buffer, alight.buffer, arrive.buffer, loadOfCrush.buffer, fareX.buffer, parts.seg.buffer, parts.board.buffer, parts.alight.buffer, ...big] };
      };
      for (const c of early) {
        setUp(c);
        for (let round = 0; round <= job.rounds && !stale; round++) {
          for (const q of mine) {
            if (c.reuse & (1 << q)) continue;
            await yieldNow();
            if (pending || !(await step(c, round, q))) { stale = true; break; }
            release_skims();
          }
        }
        if (stale) break;
        solved(c);
        // the last one's figures go with the solve's results, below
        if (c === early[early.length - 1]) break;
        for (const q of mine) {
          const { msg, transfer } = figures(q, job.rounds, 0, [c]);
          sendMessage({ kind: "result", ...msg, cluster: { cities: c.cities, market: c.market, lines: c.lines.slice() } }, transfer);
        }
      }
      if (!stale && !job.barrier) {
        // Every cluster is solved to its last round, so the solve's result for every round is its
        // final figures, worked out once: after an edit only the final is shown, and a line new
        // since the shown result takes these at round 0 (T-324).
        stageRound = job.rounds;
        for (const q of mine) roundsHeld[q] = job.rounds;
        if (mine.length) {
          for (const [id, a] of jobLocal) {
            const kept = { mask: a.solved_mask(), sums: a.sub_sums(), everyday: everyday.get(id)?.cell_modes() ?? null };
            for (let r = 0; r <= job.rounds; r++) sumsByRound.set(`${id}:${r}`, kept);
            const copy = kept.sums.slice();
            const extra = kept.everyday?.slice() ?? null;
            sendMessage({ kind: "sums", version: job.version, round: job.rounds, city: id, mask: kept.mask, sums: copy, everyday: extra }, [copy.buffer, ...(extra ? [extra.buffer] : [])]);
          }
        }
        answerWaiting();
        for (const q of mine) {
          const t0 = performance.now();
          const { msg } = figures(q, job.rounds, 0, jobClusters);
          msg.ms = msg.summary[4] = performance.now() - t0;
          for (let r = 0; r <= job.rounds; r++) {
            const m = { ...msg, summary: [r, ...msg.summary.slice(1)], seg: msg.seg.slice(), board: msg.board.slice(), alight: msg.alight.slice(), arrive: msg.arrive.slice(), loadOfCrush: msg.loadOfCrush.slice(), parts: { keys: msg.parts.keys, seg: msg.parts.seg.slice(), board: msg.parts.board.slice(), alight: msg.parts.alight.slice() }, setupMs: r === 0 ? setupMs : 0 };
            sendMessage({ kind: "result", ...m }, [m.seg.buffer, m.board.buffer, m.alight.buffer, m.arrive.buffer, m.loadOfCrush.buffer, m.parts.seg.buffer, m.parts.board.buffer, m.parts.alight.buffer]);
          }
        }
        setupMs = 0;
      }
      // with the barrier only: without it the results were sent just above
      for (let g = 0; g < groups.length && !stale; g++) {
        const late = groups[g];
        // a group solved ahead sends its cluster's figures alone; the last group's results are the
        // solve's, with the groups before it held at their last round
        const alone = g < groups.length - 1 ? late[0] : null;
        const sumsFor = jobLocal.filter(([id]) => alone ? alone.cities.includes(id) : !groups.slice(0, g).some((cs) => cs.some((c) => c.cities.includes(id))));
        stageGroup = g;
        stageRound = -1;
        roundsHeld.fill(-1);
        for (const c of late) setUp(c);
        for (let round = 0; round <= job.rounds && !stale; round++) {
          // T-327: a worker ahead holds its last round until every worker has finished it
          while (round > 0 && mine.length && !pending && !goAllows(job.version, g, round)) {
            await new Promise<void>((r) => (wake = r));
            wake = null;
          }
          if (pending) { stale = true; break; }
          stageRound = round;
          for (const q of mine) {
            await yieldNow();
            if (pending) { stale = true; break; }
            const t0 = performance.now();
            // The period is part way between rounds until every population has had it; a query in
            // between (the yields below) must not read it.
            roundsHeld[q] = -2;
            for (const c of late) if (!(c.reuse & (1 << q)) && !(await step(c, round, q))) { stale = true; break; }
            release_skims();
            if (stale) break;
            roundsHeld[q] = round;
            if (q === mine[mine.length - 1]) {
              for (const [id, a] of sumsFor) {
                const kept = { mask: a.solved_mask(), sums: a.sub_sums(), everyday: everyday.get(id)?.cell_modes() ?? null };
                sumsByRound.set(`${id}:${round}`, kept);
                // and sent unasked, ahead of this round's last result: the client holds every
                // worker's by the time it publishes the round, and the bubbles need no query to a
                // worker that is busy with the next round
                const copy = kept.sums.slice();
                const extra = kept.everyday?.slice() ?? null;
                sendMessage({ kind: "sums", version: job.version, round, city: id, mask: kept.mask, sums: copy, everyday: extra }, [copy.buffer, ...(extra ? [extra.buffer] : [])]);
              }
            }
            const { msg, transfer } = figures(q, round, performance.now() - t0, alone ? [alone] : jobClusters);
            // T-327: ahead of the result, so the client holds every worker's answer for a round by
            // the time it publishes that round
            if (q === mine[mine.length - 1]) answerWaiting();
            if (alone) sendMessage({ kind: "result", ...msg, cluster: { cities: alone.cities, market: alone.market, lines: alone.lines.slice() } }, transfer);
            else {
              sendMessage({ kind: "result", ...msg }, transfer);
              setupMs = 0;
            }
          }
        }
        if (!stale) for (const c of late) solved(c);
      }
      if (!stale) {
        sendMessage({ kind: "done", version: job.version, memBytes: wasm_memory_bytes(), clusters: [jobClusters.length - dirty.length, dirty.length], periods: [periodsKept, periodsSolved] });
      }
    }
  } catch (err) {
    sendMessage({ kind: "failed", error: String(err) });
  }
  busy = false;
  closeDeferred();
}
