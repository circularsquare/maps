// Shared-world demand worker (T-116/T-117). Owns New York local commutes,
// Philadelphia local commutes and the separate regional market. Each assigned period
// advances all populations from one frozen total load vector at every crowding round.
// Network messages supersede work between populations; queries require the same
// version AND refinement round across all this worker's periods.
//
// Packs open as they arrive (T-149): `init` starts the WASM (and may carry packs), each `open`
// adds city packs or the regional market. A solve covers the cities open when it starts.
//
// One worker of the pool has no periods (`aux`, T-119): it only sets the network up. It is never
// busy, so the questions that need no period's result (the cells, a bubble's commuters by mode
// from the sums it is handed) are answered at once, not after a busy worker's current pass.

import init, { DemandApi, EverydayApi, IntercityApi, day_factors, demand_periods, everyday_warm_up, release_skims, wasm_memory_bytes } from "../wasm/demand/sim"; // demand only (T-051)
import { CITIES, cityToMarket, cityToWorld, HOME_CITY, setWorldFrame, worldToCity, worldToMarket, type CityId } from "../game/cities";
import { EVERYDAY_KEYS, EVERYDAY_NAMES } from "../game/purposes";
import type { DemandNetwork, FromDemand, ToDemand } from "./demandProtocol";

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
let pending: { version: number; net: DemandNetwork; rounds: number } | null = null;
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
/** the populations the network in hand was solved for, and its route nodes (`transfers`, T-176) */
let jobPopulations: (DemandApi | EverydayApi | IntercityApi)[] = [];
let jobNodes = 0;

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
function openPacks(packs: { id: CityId; header: string; bin: ArrayBuffer }[], market: string | Uint8Array | undefined) {
  for (const pack of packs) {
    if (cities.has(pack.id)) continue;
    cities.set(pack.id, new DemandApi(pack.header, new Uint8Array(pack.bin)));
    if (!aux && JSON.parse(pack.header).arrays.some((a: {name: string}) => a.name === "errand_jobs")) everyday.set(pack.id, new EverydayApi(pack.header, new Uint8Array(pack.bin)));
  }
  if (market !== undefined && !aux && !intercity) {
    // T-180: the page sends the pack's bytes; decoding 8 MB here keeps it off the main thread
    intercity = new IntercityApi(typeof market === "string" ? market : new TextDecoder().decode(market));
    // T-174: without a current derived block every worker visits all zone pairs (seconds)
    if (intercity.info()[3] !== 1) console.warn("regional market pack: derived block missing or stale, derived at opening; run pipeline/intercity.py --derive");
  }
  return Object.fromEntries(packs.map(({ id }) => {
    const info = Array.from(cities.get(id)!.info());
    // Auxiliary worker only serves commute allocation; compute workers report all local demand.
    const extra = everyday.get(id)?.info();
    if (extra) {
      let outings = 0, walking = 0;
      for (let k=0;k<extra.length;k+=3) { outings += extra[k]; walking += extra[k]*extra[k+1]; }
      info[3] = (info[2]*info[3]+walking)/(info[2]+outings);
      info[2] += outings;
    }
    return [id, info];
  })) as Record<CityId, number[]>;
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
        const cityInfo = openPacks(msg.packs, msg.intercity);
        const openMs = performance.now() - t0;
        const warmMs = warmUp();
        sendMessage({ kind: "opened", cityInfo, market: msg.intercity !== undefined, openMs, warmMs, memBytes: wasm_memory_bytes() });
      } catch (err) {
        sendMessage({ kind: "failed", error: String(err) });
      }
    });
  } else if (msg.kind === "solve") {
    pending = msg;
    void opening.then(() => { if (started && !busy) run(); });
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

/** The demand views' queries (T-078). They run between periods, never inside a solve. */
function query(msg: Exclude<ToDemand, { kind: "init" | "open" | "solve" }>) {
  // A city the network in hand was not set on (opened since, T-149) answers as if not opened.
  const a = msg.kind === "cells" || jobCities.has(msg.city) ? cities.get(msg.city) : undefined;
  type WithoutCity<T> = T extends {city: CityId} ? Omit<T, "city"> : T;
  // T-285: when the question was taken up (epoch ms) and how long it took, for ?debug=1's timings
  const qStart = performance.timeOrigin + performance.now();
  const sendMessage = (reply: WithoutCity<FromDemand>, transfer: Transferable[] = []) => {
    postMessage({ ...reply, city: msg.city, qStart, qMs: performance.timeOrigin + performance.now() - qStart }, { transfer });
  };
  const live = !!a && msg.kind !== "cells" && msg.version === current && mine.every(q => roundsHeld[q] === msg.round);
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
      const kept = a && msg.version === current ? sumsByRound.get(`${msg.city}:${msg.round}`) : undefined;
      const sums = kept ? kept.sums.slice() : null;
      const extra = kept?.everyday?.slice() ?? null;
      sendMessage({ kind: "subSums", id: msg.id, version: msg.version, mask: kept ? kept.mask : 0, sums, everyday: extra }, [...(sums ? [sums.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "cellModes") {
      // reads the network's subzones and the sums it is given, not any period's state, so it holds
      // for the version whatever round this worker is in
      const modes = a && msg.version === current ? a.cell_modes_day(msg.sums, day) : null;
      sendMessage({ kind: "cellModes", id: msg.id, version: msg.version, modes: modes && modes.length ? modes : null }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "station") {
      const packed = live ? a!.station_riders_day(msg.station, day) : null;
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
      const packed = live ? a!.flows_day(msg.end, msg.cells, msg.sums, day) : null;
      sendMessage({ kind: "flows", id: msg.id, version: msg.version, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icSel") {
      // a city outside the regional market (market -1, T-277) has no longer trips
      const ok = live && !!intercity && MARKET_OF[msg.city] >= 0;
      const sq = ok ? selectionSquares(msg.city, a!, msg.end, msg.cells) : null;
      const packed = sq ? Float64Array.from(intercity!.selection_flows_day(MARKET_OF[msg.city], sq.xy, sq.share, day)) : null;
      sendMessage({ kind: "icSel", id: msg.id, version: msg.version, mask: ok ? intercity!.solved_mask() : 0, packed }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icStation") {
      const ok = live && !!intercity;
      const packed = ok ? Float64Array.from(intercity!.station_flows_day(msg.station, day)) : null;
      sendMessage({ kind: "icStation", id: msg.id, version: msg.version, mask: ok ? intercity!.solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "purposeModes") {
      // one everyday purpose's legs (T-158); a city without that purpose answers empty, not null
      const layer = live ? everyday.get(msg.city) : undefined;
      const modes = !live ? null : layer && msg.purpose < layer.purpose_count() ? layer.cell_modes_of_day(msg.purpose, day) : new Float32Array(0);
      sendMessage({ kind: "purposeModes", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, modes }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "transfers") {
      const pairs = live ? transfersAt(msg.nodes, day) : null;
      sendMessage({ kind: "transfers", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, pairs }, pairs ? [pairs.buffer] : []);
    } else if (msg.kind === "trip") {
      // T-195: the trip as a commute, each period this worker holds; the cells back in world metres
      const answers: Float64Array[] = [], periods: number[] = [];
      if (live) {
        const o = worldToCity(msg.from[0], msg.from[1], msg.city), d = worldToCity(msg.to[0], msg.to[1], msg.city);
        for (const q of mine) {
          const v = a!.trip(q, o.x, o.y, d.x, d.y);
          if (v[0] === 1) {
            const p0 = cityToWorld(v[27], v[28], msg.city), p1 = cityToWorld(v[29], v[30], msg.city);
            v[27] = p0.x; v[28] = p0.y; v[29] = p1.x; v[30] = p1.y;
          }
          answers.push(v);
          periods.push(q);
        }
      }
      sendMessage({ kind: "trip", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, periods, answers: live ? answers : null }, answers.map((v) => v.buffer));
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
  }
}

/**
 * Transfers with either end at one of `nodes` (T-176): every population of the job, every period
 * this worker holds, each (arriving, leaving) route node pair once. Asked for one station complex
 * on selection, so the answer is a few thousand numbers, never the whole network's.
 */
function transfersAt(nodes: Uint32Array, day: number): Float32Array {
  const sel = new Uint8Array(jobNodes);
  for (const r of nodes) if (r < jobNodes) sel[r] = 1;
  const sum = new Map<number, number>();
  for (const p of jobPopulations) {
    const v = p.transfers_day(sel, day);
    for (let k = 0; k + 2 < v.length; k += 3) {
      const key = v[k] * jobNodes + v[k + 1];
      sum.set(key, (sum.get(key) ?? 0) + v[k + 2]);
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
 * A period's `seg`, `board` and `alight` split by trip purpose (T-158, demandProtocol `parts`):
 * work is every city's commutes, then the everyday layer's purposes, then the regional market's
 * longer trips. They add up to the result's totals; crowding is never split.
 */
function purposeParts(q: number, nr: number, local: DemandApi[], days: EverydayApi[], regional: IntercityApi | null) {
  const kinds = Math.min(EVERYDAY_KEYS.length, Math.max(0, ...days.map((e) => e.purpose_count())));
  const keys: string[] = ["work", ...EVERYDAY_KEYS.slice(0, kinds)];
  if (regional) keys.push("longer");
  const seg = new Float32Array(keys.length * nr), board = new Float32Array(keys.length * nr), alight = new Float32Array(keys.length * nr);
  const add = (out: Float32Array, at: number, v: ArrayLike<number>, from = 0) => {
    if (v.length < from + nr) return; // nothing solved for this population (no network)
    for (let r = 0; r < nr; r++) out[at * nr + r] += v[from + r];
  };
  for (const a of local) { add(seg, 0, a.seg(q)); add(board, 0, a.board(q)); add(alight, 0, a.alight(q)); }
  for (const e of days) {
    const v = e.part_loads(q), m = Math.min(kinds, e.purpose_count());
    if (v.length !== 3 * e.purpose_count() * nr) continue;
    for (let k = 0; k < m; k++) { add(seg, 1 + k, v, 3 * k * nr); add(board, 1 + k, v, (3 * k + 1) * nr); add(alight, 1 + k, v, (3 * k + 2) * nr); }
  }
  if (regional) {
    const at = keys.length - 1;
    add(seg, at, regional.seg(q)); add(board, at, regional.board(q)); add(alight, at, regional.alight(q));
    // T-276: the regional layer's commutes between zones are work: moved from longer trips to it
    const c = regional.commute_loads(q);
    if (c.length === 3 * nr) {
      for (let r = 0; r < nr; r++) {
        seg[r] += c[r]; seg[at * nr + r] -= c[r];
        board[r] += c[nr + r]; board[at * nr + r] -= c[nr + r];
        alight[r] += c[2 * nr + r]; alight[at * nr + r] -= c[2 * nr + r];
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
      const jobLocal = [...cities];
      const jobEveryday = [...everyday].filter(([id]) => cities.has(id));
      jobCities = new Set();
      // T-161: each station's complex, for every population (a check's network may have none)
      const complex = n.complex ?? new Uint32Array(0);
      for (const [id, a] of jobLocal) {
        const xy = n.stXY.slice();
        for (let k = 0; k < xy.length; k += 2) {
          const p = worldToCity(xy[k], xy[k + 1], id); xy[k] = p.x; xy[k + 1] = p.y;
        }
        a.set_complexes(complex);
        setupMs += a.set_network(xy, n.lineN, n.stops, n.times, n.tph, n.cars);
        a.set_fares(n.fare.base, n.fare.perKm, n.hopKm);
        const extra = everyday.get(id);
        if (extra) {
          extra.set_complexes(complex);
          setupMs += extra.set_network(xy, n.lineN, n.stops, n.times, n.tph, n.cars);
          extra.set_fares(n.fare.base, n.fare.perKm, n.hopKm);
        }
        jobCities.add(id);
      }
      const regional = intercity;
      if (regional) {
        const mxy = n.stXY.slice();
        for (let k = 0; k < mxy.length; k += 2) {
          const p = worldToMarket(mxy[k], mxy[k + 1]); mxy[k] = p.x; mxy[k + 1] = p.y;
        }
        regional.set_complexes(complex);
        setupMs += regional.set_network(mxy, n.lineN, n.stops, n.times, n.tph, n.cars);
        regional.set_fares(n.fare.base, n.fare.perKm, n.hopKm);
      }
      current = job.version;
      roundsHeld.fill(-1);
      sumsByRound.clear();
      let stale = false;
      // Periods are independent. Every population in a period advances from one frozen
      // combined load vector, never from whichever population happened to run first.
      const populations: (DemandApi | IntercityApi | EverydayApi)[] = [...jobLocal.map(([, a]) => a), ...jobEveryday.map(([, a]) => a)];
      if (regional) populations.push(regional);
      jobPopulations = populations;
      jobNodes = 2 * n.stops.length;
      for (let round = 0; round <= job.rounds && !stale; round++) {
        for (const q of mine) {
          await yieldNow();
          if (pending) { stale = true; break; }
          const t0 = performance.now();
          const sum = (read: (a: DemandApi | IntercityApi | EverydayApi) => Float32Array): Float32Array => {
            const out = new Float32Array(2 * n.stops.length);
            for (const a of populations) { const v = read(a); for (let k = 0; k < out.length; k++) out[k] += v[k] ?? 0; }
            return out;
          };
          // The period is part way between rounds until every population has had it; a query in
          // between (the yields below) must not read it.
          roundsHeld[q] = -2;
          // every population's segment loads, boardings and alightings per route node: the
          // crowding and the waits for full trains (T-188) the round is solved with
          let total: Float32Array | null = null;
          if (round > 0) {
            const nr = 2 * n.stops.length;
            total = new Float32Array(3 * nr);
            total.set(sum(a => a.seg(q)), 0); total.set(sum(a => a.board(q)), nr); total.set(sum(a => a.alight(q)), 2 * nr);
          }
          for (const a of populations) {
            if (total === null) a.solve(q);
            else if (!a.crowd_with_external(q, total)) throw new Error("shared crowding layout disagrees");
            // a population's pass is up to a second of one call: let a waiting query or a newer
            // network in between
            await yieldNow();
            if (pending) { stale = true; break; }
          }
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
          const ms = performance.now() - t0;
          const seg = sum(a => a.seg(q)), board = sum(a => a.board(q)), alight = sum(a => a.alight(q)), loadOfCrush = sum(a => a.load_of_crush(q));
          const citySummary = Object.fromEntries(jobLocal.map(([id, a]) => [id, Array.from(a.summary(q))])) as Record<CityId, number[]>;
          for (const [id, a] of jobEveryday) { const v = a.summary(q); for (const k of [1, 2, 3, 5]) citySummary[id][k] += v[k] ?? 0; }
          const intercitySummary = Array.from(regional!.summary(q));
          const summary = [round, 0, 0, 0, ms, 0, 0, 0, 0, 0];
          for (const v of [...Object.values(citySummary), intercitySummary]) for (const k of [1, 2, 3, 5]) summary[k] += v[k] ?? 0;
          const parts = purposeParts(q, 2 * n.stops.length, jobLocal.map(([, a]) => a), jobEveryday.map(([, a]) => a), regional);
          // T-260: each city's trips, rail and walked trips by purpose, `parts.keys` order less the
          // market, so the main thread can work out a weekend's mode split
          const local = parts.keys.length - (regional ? 1 : 0);
          const cityParts = Object.fromEntries(jobLocal.map(([id, a]) => {
            const v = new Array(3 * local).fill(0), s = a.summary(q);
            for (let k = 0; k < 3; k++) v[k] = s[1 + k] ?? 0;
            const e = everyday.get(id)?.kind_totals(q);
            if (e) for (let k = 0; k < Math.min(e.length, 3 * (local - 1)); k++) v[3 + k] = e[k];
            return [id, v];
          })) as Record<CityId, number[]>;
          sendMessage(
            { kind: "result", version: job.version, period: q, summary, citySummary, cityParts, intercitySummary, intercityParts: Array.from(regional!.part_summary(q)), corridors: Array.from(regional!.corridor_summary(q)), commuteCorridors: Array.from(regional!.commute_corridors(q)), seg, board, alight, loadOfCrush, parts, ms, setupMs, memBytes: wasm_memory_bytes() },
            [seg.buffer, board.buffer, alight.buffer, loadOfCrush.buffer, parts.seg.buffer, parts.board.buffer, parts.alight.buffer],
          );
          setupMs = 0;
        }
      }
      if (!stale) sendMessage({ kind: "done", version: job.version, memBytes: wasm_memory_bytes() });
    }
  } catch (err) {
    sendMessage({ kind: "failed", error: String(err) });
  }
  busy = false;
}
