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

import init, { DemandApi, EverydayApi, IntercityApi, day_factors, demand_periods, everyday_warm_up, release_skims, wasm_memory_bytes } from "../wasm/demand/sim"; // demand only (T-051)
import { CITIES, cityToMarket, cityToWorld, HOME_CITY, setWorldFrame, worldToCity, worldToMarket, type CityId } from "../game/cities";
import { EVERYDAY_KEYS, EVERYDAY_NAMES } from "../game/purposes";
import { PERIODS } from "../game/periods";
import type { BareFigures, DemandNetwork, FromDemand, ToDemand } from "./demandProtocol";
import { cellBox, splitClusters, type Box, type Cluster } from "./demandClusters";

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
let intercity: IntercityApi | null = null;
let mine: number[] = [];
let pending: { version: number; net: DemandNetwork; rounds: number; barrier?: boolean; whole?: boolean; first?: CityId } | null = null;
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
type Query = Exclude<ToDemand, { kind: "init" | "open" | "solve" | "go" }>;
/** T-327: the round every worker may start, by version (a solve marked `barrier` waits for it) */
let go = { version: -1, round: 0 };
/** a paused round boundary's resume (a `go` or a newer network) */
let wake: (() => void) | null = null;
/** the questions whose answer reads the periods' state at a round (the others read none) */
const BY_ROUND = new Set<Query["kind"]>(["station", "flowRail", "purposes", "icSel", "icStation", "transfers", "trip"]);
type Population = DemandApi | EverydayApi | IntercityApi;
/** T-326: the network in hand's clusters, each with its populations (cities, their everyday
 * layers, the market: the order their loads are added in), and the whole network's route nodes */
/** `pops` in the order their figures are added up; `solveOrder` the same populations in the order
 * they are set up and solved (it changes no figure) */
type JobCluster = Cluster & { pops: Population[]; solveOrder: Population[] };
let jobClusters: JobCluster[] = [];
let jobNodes = 0;
/** T-326: the cluster `sig` each population holds every one of this worker's periods solved for,
 * to the last round */
const held = new Map<Population, string>();
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
function openPacks(packs: { id: CityId; header: string; bin: ArrayBuffer }[], market: string | Uint8Array | undefined, isBare = false) {
  for (const pack of packs) {
    if (isBare) {
      if (!bare.has(pack.id)) bare.set(pack.id, openBare(pack));
      continue;
    }
    if (cities.has(pack.id)) continue;
    cities.set(pack.id, new DemandApi(pack.header, new Uint8Array(pack.bin)));
    if (!aux && hasEveryday(pack.header)) everyday.set(pack.id, new EverydayApi(pack.header, new Uint8Array(pack.bin)));
  }
  if (market !== undefined && !aux && !intercity) {
    // T-180: the page sends the pack's bytes; decoding 8 MB here keeps it off the main thread
    intercity = new IntercityApi(typeof market === "string" ? market : new TextDecoder().decode(market));
    // T-174: without a current derived block every worker visits all zone pairs (seconds)
    if (intercity.info()[3] !== 1) console.warn("regional market pack: derived block missing or stale, derived at opening; run pipeline/intercity.py --derive");
  }
  return Object.fromEntries(packs.map(({ id }) => {
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
  const e = hasEveryday(pack.header) ? new EverydayApi(pack.header, bin) : null;
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
    if (v) for (const k of [1, 2, 3, 5]) s[k] += v[k] ?? 0;
    const p = new Array(3 * keys.length).fill(0);
    for (let k = 0; k < 3; k++) p[k] = own[1 + k] ?? 0;
    const kt = e?.kind_totals(q);
    if (kt) for (let k = 0; k < Math.min(kt.length, 3 * (keys.length - 1)); k++) p[3 + k] = kt[k];
    figures.summary.push(s);
    figures.parts.push(p);
  }
  return { a, e, figures };
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
        await init();
        const t1 = performance.now();
        aux = !!msg.aux;
        mine = msg.periods;
        groupMasks = (msg.groups ?? []).map((g) => g.reduce((m, q) => m | (1 << q), 0));
        // T-173: one worker opens a small synthetic everyday layer first, so the browser has
        // optimised that code before the real cities open in every worker at once
        // (150: 22,500 cells, 900 zones; about 150 ms, mostly while the packs download)
        const everydayWarmMs = msg.warmEveryday ? everyday_warm_up(150) : 0;
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
    // between populations (its job keeps the cities it started with)
    opening = opening.then(() => {
      if (!started) return;
      try {
        const t0 = performance.now();
        const isBare = !!msg.bare && aux;
        const cityInfo = openPacks(msg.packs, msg.intercity, isBare);
        const openMs = performance.now() - t0;
        const warmMs = warmUp();
        const figures = isBare ? Object.fromEntries(msg.packs.map(({ id }) => [id, bare.get(id)!.figures])) : undefined;
        sendMessage({ kind: "opened", cityInfo, market: msg.intercity !== undefined, openMs, warmMs, memBytes: wasm_memory_bytes(), ...(figures ? { bare: figures } : {}) });
      } catch (err) {
        sendMessage({ kind: "failed", error: String(err) });
      }
    });
  } else if (msg.kind === "solve") {
    pending = msg;
    wake?.();
    void opening.then(() => { if (started && !busy) run(); });
  } else if (msg.kind === "go") {
    if (msg.version > go.version || (msg.version === go.version && msg.round > go.round)) go = { version: msg.version, round: msg.round };
    wake?.();
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
/** The market (IntercityApi's) of a local city. */
const MARKET_OF: Record<CityId, number> = Object.fromEntries(CITIES.map(c => [c.id, c.market]));
const CITY_ORDER = new Map(CITIES.map((c, i) => [c.id, i]));

/** The demand views' queries (T-078). They run between periods, never inside a solve. */
function query(msg: Query) {
  if ((msg as { wait?: boolean }).wait && msg.kind !== "cells" && BY_ROUND.has(msg.kind) && !aux) {
    const { version, round } = msg;
    const behind = version > current || (version === current && busy && (stageRound < round || (stageRound === round && !mine.every((q) => roundsHeld[q] === round))));
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
  const live = !!a && msg.kind !== "cells" && ((msg.version === current && mine.every(q => roundsHeld[q] === msg.round)) || (kept && msg.round === at!.rounds));
  // the network's subzones, whatever round (cellModes, flows)
  const sameNet = msg.kind !== "cells" && (msg.version === current || kept);
  // the regional market's state at the round asked about (its cluster may not be the city's)
  const mc = at?.clusters.find((c) => c.market);
  const marketLive = msg.kind !== "cells" && ((msg.version === current && mine.every(q => roundsHeld[q] === msg.round)) || (!!mc && isHeld(mc) && msg.round === at!.rounds));
  // T-260: the kind of day asked about (0 weekday, 1 weekend); every answer weighs its periods by it
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
      // a city outside the regional market (market -1, T-277) has no longer trips. T-336: a zone
      // open bare in the aux worker comes with its squares (this worker does not hold the zone)
      const ok = marketLive && (!!msg.sq || !!a) && !!intercity && MARKET_OF[msg.city] >= 0;
      const sq = !ok ? null : msg.sq ?? selectionSquares(msg.city, a!, msg.end, msg.cells);
      const packed = sq ? Float64Array.from(intercity!.selection_flows_day(MARKET_OF[msg.city], sq.xy, sq.share, day)) : null;
      sendMessage({ kind: "icSel", id: msg.id, version: msg.version, mask: ok ? intercity!.solved_mask() : 0, packed }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icStation") {
      const inMarket = (mc?.stLocal[msg.station] ?? -1) >= 0;
      const ok = live && (!inMarket || marketLive) && !!intercity;
      const packed = ok ? Float64Array.from(marketStationFlows(at!, intercity!, msg.station, day)) : null;
      sendMessage({ kind: "icStation", id: msg.id, version: msg.version, mask: ok ? intercity!.solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "purposeModes") {
      // one everyday purpose's legs (T-158); a city without that purpose answers empty, not null
      const layer = live ? everyday.get(msg.city) : undefined;
      const modes = !live ? null : layer && msg.purpose < layer.purpose_count() ? layer.cell_modes_of_day(msg.purpose, day) : new Float32Array(0);
      sendMessage({ kind: "purposeModes", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, modes }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "transfers") {
      const pairs = live ? transfersAt(at!, msg.version === current && mine.every(q => roundsHeld[q] === msg.round), msg.nodes, day) : null;
      sendMessage({ kind: "transfers", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, pairs }, pairs ? [pairs.buffer] : []);
    } else if (msg.kind === "trip") {
      // T-195: the trip as a commute, each period this worker holds; the cells back in world metres
      const answers: Float64Array[] = [], periods: number[] = [];
      if (live) {
        const o = worldToCity(msg.from[0], msg.from[1], msg.city), d = worldToCity(msg.to[0], msg.to[1], msg.city);
        const c = cl;
        for (const q of mine) {
          const v = a!.trip(q, o.x, o.y, d.x, d.y);
          if (v[0] === 1) {
            const p0 = cityToWorld(v[27], v[28], msg.city), p1 = cityToWorld(v[29], v[30], msg.city);
            v[27] = p0.x; v[28] = p0.y; v[29] = p1.x; v[30] = p1.y;
            if (c) tripNodesGlobal(v, c.nodeGlobal);
          }
          answers.push(v);
          periods.push(q);
        }
      }
      sendMessage({ kind: "trip", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, periods, answers: live ? answers : null }, answers.map((v) => v.buffer));
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
      const sq = b && MARKET_OF[msg.city] >= 0 ? selectionSquares(msg.city, b.a, msg.end, msg.cells) : null;
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
    // T-392: a cluster mid-solve has no figures for this network yet
    if (!allLive && !isHeld(c)) continue;
    const sel = new Uint8Array(c.nodeGlobal.length);
    let any = false;
    for (const r of nodes) {
      const l = r < jobNodes ? c.nodeLocal[r] : -1;
      if (l >= 0) (sel[l] = 1), (any = true);
    }
    if (!any) continue;
    for (const p of c.pops) {
      const v = p.transfers_day(sel, day);
      for (let k = 0; k + 2 < v.length; k += 3) {
        const key = c.nodeGlobal[v[k]] * jobNodes + c.nodeGlobal[v[k + 1]];
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
 * T-326: a station's riders from one city, on its cluster's network. A station outside the cluster
 * gets the answer a population set on the whole network gives for a station it cannot reach: no
 * riders, every zone listed (`DemandApi.station_riders_day`).
 */
function stationRiders(at: Version, c: JobCluster | undefined, a: DemandApi, station: number, day: number): Float32Array {
  const ls = c && c.net.lineN.length ? c.stLocal[station] ?? -1 : -1;
  if (ls >= 0) return a.station_riders_day(ls, day);
  return at.railed && station < at.stations ? new Float32Array(4 + 2 * a.info()[1]) : new Float32Array(0);
}
/** The same for the market's longer trips (`IntercityApi.station_flows_day`). */
function marketStationFlows(at: Version, m: IntercityApi, station: number, day: number): Float64Array {
  const c = at.clusters.find((c) => c.market);
  const ls = c ? c.stLocal[station] ?? -1 : -1;
  if (ls >= 0) return m.station_flows_day(ls, day);
  return station < at.stations ? m.no_station_flows() : new Float64Array(0);
}
/** A `DemandApi.trip` answer's route nodes (trip.ts `PAIR_LEN`, `LEG_LEN`) from its cluster's
 * network to the whole network's, the ones the client reads. */
function tripNodesGlobal(v: Float64Array, nodeGlobal: Uint32Array) {
  const HEAD = 32, PAIR_LEN = 14, LEG_LEN = 12;
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

/**
 * A period's `seg`, `board` and `alight` split by trip purpose (T-158, demandProtocol `parts`):
 * work is every city's commutes, then the everyday layer's purposes, then the regional market's
 * longer trips. They add up to the result's totals; crowding is never split. Each cluster's route
 * nodes go to their place in the whole network's (T-326).
 */
function purposeParts(q: number, nr: number, clusters: typeof jobClusters, regional: IntercityApi | null) {
  const days = clusters.flatMap((c) => c.pops.filter((p): p is EverydayApi => p instanceof EverydayApi));
  const kinds = Math.min(EVERYDAY_KEYS.length, Math.max(0, ...days.map((e) => e.purpose_count())));
  const keys: string[] = ["work", ...EVERYDAY_KEYS.slice(0, kinds)];
  if (regional) keys.push("longer");
  const seg = new Float32Array(keys.length * nr), board = new Float32Array(keys.length * nr), alight = new Float32Array(keys.length * nr);
  for (const c of clusters) {
    const map = c.nodeGlobal, m = map.length;
    const add = (out: Float32Array, at: number, v: ArrayLike<number>, from = 0) => {
      if (v.length < from + m) return; // nothing solved for this population (no network)
      for (let r = 0; r < m; r++) out[at * nr + map[r]] += v[from + r];
    };
    for (const a of c.pops) if (a instanceof DemandApi) { add(seg, 0, a.seg(q)); add(board, 0, a.board(q)); add(alight, 0, a.alight(q)); }
    for (const e of c.pops) {
      if (!(e instanceof EverydayApi)) continue;
      const v = e.part_loads(q), k1 = Math.min(kinds, e.purpose_count());
      if (v.length !== 3 * e.purpose_count() * m) continue;
      for (let k = 0; k < k1; k++) { add(seg, 1 + k, v, 3 * k * m); add(board, 1 + k, v, (3 * k + 1) * m); add(alight, 1 + k, v, (3 * k + 2) * m); }
    }
    if (regional && c.pops.includes(regional)) {
      const at = keys.length - 1;
      add(seg, at, regional.seg(q)); add(board, at, regional.board(q)); add(alight, at, regional.alight(q));
      // T-276: the regional layer's commutes between zones are work: moved from longer trips to it
      const v = regional.commute_loads(q);
      if (v.length === 3 * m) {
        for (let r = 0; r < m; r++) {
          const g = map[r];
          seg[g] += v[r]; seg[at * nr + g] -= v[r];
          board[g] += v[m + r]; board[at * nr + g] -= v[m + r];
          alight[g] += v[2 * m + r]; alight[at * nr + g] -= v[2 * m + r];
        }
      }
    }
  }
  return { keys, seg, board, alight };
}

async function run() {
  busy = true;
  try {
    while (pending && started) {
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
      const split = splitClusters(n, jobLocal.map(([id]) => ({ id, box: boxes.get(id) ?? null })), !!regional, job.rounds, job.whole);
      jobClusters = split.map((c) => ({ ...c, pops: [
        ...c.cities.map((id) => cities.get(id)!),
        ...jobEveryday.filter(([id]) => c.cities.includes(id)).map(([, e]) => e),
        ...(c.market && regional ? [regional] : []),
      ], solveOrder: [
        // the order the packs opened in (the viewed zone's first); the order changes no figure. Set
        // up in registry order instead, the real save's first round-0 solve on each worker is ~1 s
        // slower in WASM (T-405, notes/T-405.md; the skim cache no longer depends on the order)
        ...[...cities.keys()].filter((id) => c.cities.includes(id)).map((id) => cities.get(id)!),
        ...[...everyday.keys()].filter((id) => c.cities.includes(id) && jobCities.has(id)).map((id) => everyday.get(id)!),
        ...(c.market && regional ? [regional] : []),
      ] }));
      const dirty = jobClusters.filter((c) => c.pops.some((p) => held.get(p) !== c.sig));
      // T-393: the cluster of the city in view first
      if (job.first) {
        const i = dirty.findIndex((c) => c.cities.includes(job.first!));
        if (i > 0) dirty.unshift(...dirty.splice(i, 1));
      }
      // T-392: after an edit (no barrier) the clusters are solved one after another, each to its last
      // round, and each but the last is sent as a cluster result the moment it is done; then the
      // solve's results. With the barrier (the shown estimates step through the rounds) they all go
      // round by round together.
      const early = job.barrier ? [] : dirty;
      const late = job.barrier ? dirty : [];
      const setUp = (c: JobCluster) => {
        const m = c.net;
        for (const p of c.solveOrder) {
          held.delete(p);
          const city = p === regional ? null : c.cities.find((id) => cities.get(id) === p || everyday.get(id) === p)!;
          const xy = m.stXY.slice();
          for (let k = 0; k < xy.length; k += 2) {
            const w = city ? worldToCity(xy[k], xy[k + 1], city) : worldToMarket(xy[k], xy[k + 1]);
            xy[k] = w.x; xy[k + 1] = w.y;
          }
          p.set_complexes(m.complex);
          setupMs += p.set_network(xy, m.lineN, m.stops, m.times, m.tph, m.cars);
          p.set_fares(m.fare.base, m.fare.perKm, m.hopKm);
        }
      };
      jobNodes = 2 * n.stops.length;
      versions.set(job.version, { clusters: jobClusters, stations: n.stXY.length / 2, railed: n.stXY.length > 0 && n.lineN.length > 0, nodes: jobNodes, rounds: job.rounds });
      for (const v of versions.keys()) if (versions.size > KEEP_VERSIONS && v !== job.version) versions.delete(v);
      sendMessage({ kind: "plan", version: job.version, solving: dirty.map((c) => c.cities), kept: jobClusters.filter((c) => !dirty.includes(c)).flatMap((c) => c.cities) });
      current = job.version;
      roundsHeld.fill(-1);
      sumsByRound.clear();
      stageRound = -1;
      answerWaiting(); // an older version's get nulls; this one's wait on
      let stale = false;
      // a cluster's loads, all its populations', per route node of its network
      const sum = (c: JobCluster, read: (a: Population) => Float32Array): Float32Array => {
        const out = new Float32Array(c.nodeGlobal.length);
        for (const a of c.pops) { const v = read(a); for (let k = 0; k < out.length; k++) out[k] += v[k] ?? 0; }
        return out;
      };
      // the clusters', per route node of the whole network
      const sumAll = (cs: JobCluster[], read: (a: Population) => Float32Array): Float32Array => {
        const out = new Float32Array(jobNodes);
        for (const c of cs) {
          const map = c.nodeGlobal;
          for (const a of c.pops) { const v = read(a); for (let k = 0; k < map.length; k++) out[map[k]] += v[k] ?? 0; }
        }
        return out;
      };
      // One round of period q for the cluster. Periods are independent. Every population of a
      // cluster advances from one frozen load vector of the whole cluster, never from whichever
      // population happened to run first. False if a newer network came meanwhile.
      const step = async (c: JobCluster, round: number, q: number): Promise<boolean> => {
        // the cluster's segment loads, boardings and alightings per route node: the crowding and
        // the waits for full trains (T-188) the round is solved with
        let total: Float32Array | null = null;
        if (round > 0) {
          const nr = c.nodeGlobal.length;
          total = new Float32Array(3 * nr);
          total.set(sum(c, a => a.seg(q)), 0); total.set(sum(c, a => a.board(q)), nr); total.set(sum(c, a => a.alight(q)), 2 * nr);
        }
        for (const a of c.solveOrder) {
          if (total === null) a.solve(q);
          else if (!a.crowd_with_external(q, total)) throw new Error("shared crowding layout disagrees");
          // a population's pass is up to a second of one call: let a waiting query or a newer
          // network in between
          await yieldNow();
          if (pending) return false;
        }
        return true;
      };
      // period q's figures over the clusters `cs` (all of them for a result), as the state stands
      const figures = (q: number, round: number, ms: number, cs: JobCluster[]) => {
        // in the job's order: the summary adds the cities up in it
        const ids = jobLocal.map(([id]) => id).filter((id) => cs.some((c) => c.cities.includes(id)));
        const market = regional && cs.some((c) => c.market) ? regional : null;
        const seg = sumAll(cs, a => a.seg(q)), board = sumAll(cs, a => a.board(q)), alight = sumAll(cs, a => a.alight(q)), loadOfCrush = sumAll(cs, a => a.load_of_crush(q));
        const citySummary = Object.fromEntries(ids.map((id) => [id, Array.from(cities.get(id)!.summary(q))])) as Record<CityId, number[]>;
        for (const [id, a] of jobEveryday) if (citySummary[id]) { const v = a.summary(q); for (const k of [1, 2, 3, 5]) citySummary[id][k] += v[k] ?? 0; }
        const intercitySummary = market ? Array.from(market.summary(q)) : [];
        const summary = [round, 0, 0, 0, ms, 0, 0, 0, 0, 0];
        for (const v of [...Object.values(citySummary), intercitySummary]) for (const k of [1, 2, 3, 5]) summary[k] += v[k] ?? 0;
        const parts = purposeParts(q, jobNodes, cs, market);
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
        const msg = { version: job.version, period: q, summary, citySummary, cityParts, intercitySummary, intercityParts: market ? Array.from(market.part_summary(q)) : [], corridors: market ? Array.from(market.corridor_summary(q)) : [], commuteCorridors: market ? Array.from(market.commute_corridors(q)) : [], seg, board, alight, loadOfCrush, parts, ms, setupMs, memBytes: wasm_memory_bytes() };
        return { msg, transfer: [seg.buffer, board.buffer, alight.buffer, loadOfCrush.buffer, parts.seg.buffer, parts.board.buffer, parts.alight.buffer] };
      };
      for (const c of early) {
        setUp(c);
        for (let round = 0; round <= job.rounds && !stale; round++) {
          for (const q of mine) {
            await yieldNow();
            if (pending || !(await step(c, round, q))) { stale = true; break; }
            release_skims();
          }
        }
        if (stale) break;
        for (const p of c.pops) held.set(p, c.sig);
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
            const m = { ...msg, summary: [r, ...msg.summary.slice(1)], seg: msg.seg.slice(), board: msg.board.slice(), alight: msg.alight.slice(), loadOfCrush: msg.loadOfCrush.slice(), parts: { keys: msg.parts.keys, seg: msg.parts.seg.slice(), board: msg.parts.board.slice(), alight: msg.parts.alight.slice() }, setupMs: r === 0 ? setupMs : 0 };
            sendMessage({ kind: "result", ...m }, [m.seg.buffer, m.board.buffer, m.alight.buffer, m.loadOfCrush.buffer, m.parts.seg.buffer, m.parts.board.buffer, m.parts.alight.buffer]);
          }
        }
        setupMs = 0;
      }
      if (!stale) for (const c of late) setUp(c);
      // with the barrier only: without it the results were sent just above
      for (let round = 0; job.barrier && round <= job.rounds && !stale; round++) {
        // T-327: a worker ahead holds its last round until every worker has finished it
        while (job.barrier && round > 0 && !pending && !(go.version > job.version || (go.version === job.version && go.round >= round))) {
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
          for (const c of late) if (!(await step(c, round, q))) { stale = true; break; }
          release_skims();
          if (stale) break;
          roundsHeld[q] = round;
          if (q === mine[mine.length - 1]) {
            for (const [id, a] of jobLocal) {
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
          const { msg, transfer } = figures(q, round, performance.now() - t0, jobClusters);
          // T-327: ahead of the result, so the client holds every worker's answer for a round by
          // the time it publishes that round
          if (q === mine[mine.length - 1]) answerWaiting();
          sendMessage({ kind: "result", ...msg }, transfer);
          setupMs = 0;
        }
      }
      if (!stale) {
        for (const c of late) for (const p of c.pops) held.set(p, c.sig);
        sendMessage({ kind: "done", version: job.version, memBytes: wasm_memory_bytes(), clusters: [jobClusters.length - dirty.length, dirty.length] });
      }
    }
  } catch (err) {
    sendMessage({ kind: "failed", error: String(err) });
  }
  busy = false;
}
