// Main-thread side of the demand worker pool (T-026, T-021; SPEC 4.4, 6.5, 7; notes/T-026.md).
//
// - Fetches the city pack once and hands each worker a copy; each worker parses it and keeps its
//   own city (workers own their memory, no SharedArrayBuffer).
// - The day's five periods are split over the pool by index (worker i solves periods i, i + n,
//   ...). Each worker sends the free-flow estimate of all its periods first, then the crowding
//   rounds.
// - `submitNetwork` takes the clock worker's NetworkSnapshots (already debounced there, SPEC 6.5).
//   With `?demandTest=1` the pool solves T-005's hand-made network instead (workers/demandTest.ts)
//   and ignores the clock's.
// - Never blocks: a newer network supersedes a running solve at the workers' next period
//   boundary; game/demand.ts keeps the last finished result until every period of the new one
//   has landed, then swaps it in whole (free flow first, again after each crowding round).

import {
  baseSplit,
  demandUpdating,
  demandView,
  setDemandQueries,
  type CellModes,
  type CellRiders,
  type CityCells,
  type Flows,
  type LineDemand,
  type StationDemand,
  type StationRiders,
  type ZoneRiders,
} from "../game/demand";
import { PERIODS, periodsMatch } from "../game/periods";
import type { DemandNetwork, FromDemand, ToDemand } from "./demandProtocol";
import { demandTestSnapshot } from "./demandTest";
import type { NetworkSnapshot } from "./protocol";

/**
 * Crowding rounds after the free-flow estimate (T-022, notes/T-022.md). Each lands as its own
 * update. One round leaves the worst segment at its most overloaded (the first round moves half
 * the riders a penalty pushes, all at once); by the third a player-sized network moves under 1% a
 * round and New York's real network under 10%.
 */
const ROUNDS = 3;
const PERIOD_STRIDE = 6;

/**
 * Workers in the pool: half the logical cores less one, at most 3, at least 1. Three workers
 * solve the five periods in two rounds of one period each (about as fast as five would on a
 * busy machine) and leave the rest of a desktop for the browser, the clock worker and whatever
 * else is running; a 4-core laptop gets one. `?demandWorkers=N` (1-5) overrides it.
 */
export function poolSize(): number {
  const q = Number(new URLSearchParams(location.search).get("demandWorkers"));
  if (Number.isInteger(q) && q >= 1 && q <= PERIODS.length) return q;
  const hc = navigator.hardwareConcurrency || 4;
  return Math.max(1, Math.min(3, Math.floor(hc / 2) - 1));
}

/** The service demand reads, with the UI's ids. */
interface Service {
  fare: { base: number; perKm: number };
  stations: { id: string; x: number; y: number }[];
  lines: { id: string; stops: string[]; tph: [number, number, number]; hopS: number[][][]; hopKm?: number[][]; dwellS: number[][][]; cars: number }[];
}

type Result = Extract<FromDemand, { kind: "result" }>;
type Reply = Extract<FromDemand, { kind: "cells" | "subSums" | "cellModes" | "station" | "flowRail" | "flows" }>;
type Query = Extract<ToDemand, { kind: "cells" | "subSums" | "cellModes" | "station" | "flowRail" | "flows" }>;
const ALL_PERIODS = (1 << PERIODS.length) - 1;

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
  published: number;
  /** performance.now(): the network arrived, was sent to the workers, the free-flow estimate and
   * the refined result were published */
  arrivedAt: number;
  sentAt: number;
  firstAt?: number;
  refinedAt?: number;
  periodMs: number[][];
  setupMs: number[];
  /** WASM linear memory per worker after its last stage, bytes */
  memBytes: number[];
}

function serviceFromSnapshot(s: NetworkSnapshot): Service {
  return {
    fare: s.fare ?? { base: 2, perKm: 0 }, // legacy fixed 6 perceived minutes
    stations: s.stations.map((st) => ({ id: String(st.id), x: st.x, y: st.y })),
    lines: s.lines.map((l) => ({ id: String(l.id), stops: l.stops.map(String), tph: l.tph, hopS: l.hopS, hopKm: l.hopKm, dwellS: l.dwellS, cars: l.cars })),
  };
}

/** Pack a service for the workers; lines naming a station that is not in the list are left out. */
function packService(svc: Service): { net: DemandNetwork; layout: Solve["layout"] } {
  const index = new Map(svc.stations.map((s, i) => [s.id, i]));
  const lines = svc.lines.filter((l) => l.stops.length >= 2 && l.stops.every((s) => index.has(s)));
  const nStops = lines.reduce((a, l) => a + l.stops.length, 0);
  const net: DemandNetwork = {
    stXY: new Float32Array(svc.stations.flatMap((s) => [s.x, s.y])),
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
  private workers: Worker[] = [];
  private periodsOf: number[][] = [];
  private ready = 0;
  /** home-to-work trips a day, from the first worker's info */
  private info: number[] | null = null;
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
  private onReady: (() => void)[] = [];
  private cellsP: Promise<CityCells | null> | null = null;
  private modesP: { version: number; round: number; p: Promise<CellModes | null> } | null = null;
  readonly start: { packMs?: number; postMs?: number; workers: { wasmMs: number; openMs: number; warmMs: number; memBytes: number }[] } = { workers: [] };
  status = "loading";

  constructor(city: string) {
    const n = poolSize();
    for (let i = 0; i < n; i++) {
      const w = new Worker(new URL("./demand.worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<FromDemand>) => this.receive(i, e.data);
      w.onerror = (e) => this.fail(`worker ${i}: ${e.message}`);
      this.workers.push(w);
      this.periodsOf.push(PERIODS.map((_, q) => q).filter((q) => q % n === i));
    }
    this.load(city);
  }

  private fail(why: string) {
    this.status = "failed: " + why;
    demandUpdating.value = false;
    console.error("demand:", why);
  }

  private async load(city: string) {
    const t0 = performance.now();
    try {
      const [header, bin] = await Promise.all([
        fetch(new URL(`packs/${city}.json`, document.baseURI)).then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${r.status} ${r.url}`)))),
        fetch(new URL(`packs/${city}.bin`, document.baseURI)).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${r.status} ${r.url}`)))),
      ]);
      this.start.packMs = performance.now() - t0;
      const t1 = performance.now();
      // each worker gets its own buffer, moved rather than copied again: a copy for every
      // worker but the last, which gets the original
      this.workers.forEach((w, i) => {
        const own = i === this.workers.length - 1 ? bin : bin.slice(0);
        const msg: ToDemand = { kind: "init", header, bin: own, periods: this.periodsOf[i] };
        w.postMessage(msg, [own]);
      });
      this.start.postMs = performance.now() - t1;
    } catch (err) {
      this.fail("city pack: " + String(err));
    }
  }

  private receive(i: number, msg: FromDemand) {
    if (msg.kind === "failed") return this.fail(`worker ${i}: ${msg.error}`);
    if (msg.kind === "cells" || msg.kind === "subSums" || msg.kind === "cellModes" || msg.kind === "station" || msg.kind === "flowRail" || msg.kind === "flows") {
      const w = this.pendingQueries.get(msg.id);
      if (!w) return;
      w.replies.push(msg);
      if (--w.left === 0) {
        this.pendingQueries.delete(msg.id);
        w.done(w.replies);
      }
      return;
    }
    if (msg.kind === "ready") {
      this.start.workers[i] = { wasmMs: msg.wasmMs, openMs: msg.openMs, warmMs: msg.warmMs, memBytes: msg.memBytes };
      if (!periodsMatch(msg.periods, PERIOD_STRIDE))
        console.error("demand: game/periods.ts disagrees with sim/src/track/params.rs PERIOD_HOURS / PERIOD_LEVEL", msg.periods);
      if (!this.info) {
        this.info = msg.info;
        this.periodTable = msg.periods;
        const walk = 100 * msg.info[3];
        if (!demandView.peek()) baseSplit.value = { train: 0, walk, drive: 100 - walk };
      }
      if (++this.ready === this.workers.length) {
        this.status = "ready";
        this.onReady.splice(0).forEach((f) => f());
        if (this.waiting) {
          const w = this.waiting;
          this.waiting = null;
          this.send(w.svc, w.arrivedAt);
        }
      }
      return;
    }
    const s = this.cur;
    if (!s || msg.version !== s.version) return; // superseded
    if (msg.kind === "done") return;
    s.results[msg.period] = msg;
    (s.periodMs[msg.period] ??= [])[msg.summary[0]] = msg.ms;
    if (msg.setupMs) s.setupMs[i] = msg.setupMs;
    s.memBytes[i] = msg.memBytes;
    const round = Math.min(...s.results.map((r) => (r ? r.summary[0] : -1)));
    if (round > s.published) {
      s.published = round;
      this.publish(s, round);
    }
  }

  /** A network from the clock worker (T-025), or the test network. */
  submitNetwork(snap: NetworkSnapshot) {
    const svc = serviceFromSnapshot(snap);
    const key = JSON.stringify(svc);
    if (key === this.lastKey) return; // nothing demand reads changed
    this.lastKey = key;
    demandUpdating.value = true;
    this.send(svc, performance.now());
  }

  /** Solve the current network again under a new version (timing runs). */
  resubmit() {
    if (this.cur) this.send(this.cur.svc, performance.now());
  }

  private send(svc: Service, arrivedAt: number) {
    if (this.ready < this.workers.length) {
      this.waiting = { svc, arrivedAt };
      return;
    }
    const { net, layout } = packService(svc);
    const version = ++this.version;
    this.cur = { version, svc, layout, results: PERIODS.map(() => null), published: -1, arrivedAt, sentAt: performance.now(), periodMs: [], setupMs: [], memBytes: [] };
    this.log.push(this.cur);
    if (this.log.length > 50) this.log.shift();
    demandUpdating.value = true;
    const msg: ToDemand = { kind: "solve", version, net, rounds: ROUNDS };
    for (const w of this.workers) w.postMessage(msg);
  }

  // --- Demand views (T-078; types and meaning in game/demand.ts) ---------------------------------

  /** Send `make(id)` to the given workers (all by default) and collect their replies. */
  private ask(make: (id: number) => Query, which: number[] = this.workers.map((_, i) => i), transfer: Transferable[] = []): Promise<Reply[]> {
    const id = this.nextQuery++;
    return new Promise((done) => {
      this.pendingQueries.set(id, { left: which.length, replies: [], done });
      for (const i of which) this.workers[i].postMessage(make(id), transfer);
    });
  }

  private whenReady(): Promise<boolean> {
    if (this.status.startsWith("failed")) return Promise.resolve(false);
    if (this.ready === this.workers.length) return Promise.resolve(true);
    return new Promise((r) => this.onReady.push(() => r(true)));
  }

  cityCells(): Promise<CityCells | null> {
    return (this.cellsP ??= this.whenReady().then(async (ok) => {
      if (!ok) return null;
      const [r] = await this.ask((id) => ({ kind: "cells", id }), [0]);
      if (r.kind !== "cells") return null;
      return { n: r.xy.length / 2, xy: r.xy, h3: r.h3, zone: r.zone, zoneM: r.zoneXY[0], zoneXY: r.zoneXY.subarray(1) };
    }));
  }

  cellModes(): Promise<CellModes | null> {
    const version = demandView.peek()?.version;
    if (version === undefined) return Promise.resolve(null);
    // cached per solve and crowding round (T-084: the dot map asks again for the refined result)
    const round = demandView.peek()?.round ?? 0;
    if (this.modesP?.version === version && this.modesP.round === round) return this.modesP.p;
    const p = (async (): Promise<CellModes | null> => {
      const parts = await this.ask((id) => ({ kind: "subSums", id, version }));
      let mask = 0;
      let sums: Float32Array | null = null;
      for (const r of parts) {
        if (r.kind !== "subSums" || !r.sums) return null;
        mask |= r.mask;
        if (!sums) sums = r.sums;
        else for (let k = 0; k < sums.length; k++) sums[k] += r.sums[k];
      }
      if (!sums || mask !== ALL_PERIODS) return null;
      const [r] = await this.ask((id) => ({ kind: "cellModes", id, version, sums: sums! }), [0], [sums.buffer]);
      if (r.kind !== "cellModes" || !r.modes) return null;
      const n = r.modes.length / 6;
      const b = (k: number) => r.modes!.subarray(k * n, (k + 1) * n);
      return { version, homeRail: b(0), homeWalk: b(1), homeDrive: b(2), workRail: b(3), workWalk: b(4), workDrive: b(5) };
    })();
    this.modesP = { version, round, p };
    p.then((v) => {
      if (!v && this.modesP?.p === p) this.modesP = null; // not cached when it failed
    });
    return p;
  }

  async stationRiders(stationId: string, top = 20): Promise<StationRiders | null> {
    const version = demandView.peek()?.version;
    const solve = this.log.find((s) => s.version === version);
    if (version === undefined || !solve) return null;
    const station = solve.svc.stations.findIndex((s) => s.id === stationId);
    if (station < 0) return null;
    const parts = await this.ask((id) => ({ kind: "station", id, version, station }));
    let mask = 0;
    const home = new Map<number, number>(), work = new Map<number, number>();
    let toZone: Float32Array | null = null, fromZone: Float32Array | null = null;
    for (const r of parts) {
      if (r.kind !== "station" || !r.packed) return null;
      mask |= r.mask;
      const p = r.packed;
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
    if (mask !== ALL_PERIODS || !toZone || !fromZone) return null;
    const cells = (m: Map<number, number>): CellRiders => {
      const t = topOf([...m.keys()], [...m.values()]);
      return { cells: t.keys, riders: t.vals, total: t.vals.reduce((a, b) => a + b, 0) };
    };
    const zones = (v: Float32Array): ZoneRiders => {
      const t = topOf(Uint32Array.from(v, (_, i) => i), v, top);
      return { zones: t.keys, riders: t.vals };
    };
    return { version, stationId, home: cells(home), work: cells(work), homeRidersWorkIn: zones(toZone), workRidersLiveIn: zones(fromZone) };
  }

  /** Where the commuters of a set of cells go (T-098; game/demand.ts `Flows`): every worker's rail
   * part of its periods, added up, then one worker spreads all modes over the far end. */
  async flows(end: "home" | "work", cells: Uint32Array): Promise<Flows | null> {
    const version = demandView.peek()?.version;
    if (version === undefined || cells.length === 0) return null;
    const e = end === "home" ? 0 : 1;
    const parts = await this.ask((id) => ({ kind: "flowRail", id, version, end: e, cells }));
    let mask = 0;
    let sums: Float32Array | null = null;
    for (const r of parts) {
      if (r.kind !== "flowRail" || !r.sums) return null;
      mask |= r.mask;
      if (!sums) sums = r.sums;
      else for (let k = 0; k < sums.length; k++) sums[k] += r.sums[k];
    }
    if (!sums || mask !== ALL_PERIODS) return null;
    const [r] = await this.ask((id) => ({ kind: "flows", id, version, end: e, cells, sums: sums! }), [0], [sums.buffer]);
    if (r.kind !== "flows" || !r.packed) return null;
    const p = r.packed;
    const n = p[6];
    const at = (k: number) => p.subarray(7 + k * n, 7 + (k + 1) * n);
    return {
      version,
      end,
      rail: p[0],
      walk: p[1],
      drive: p[2],
      cutRail: p[3],
      cutWalk: p[4],
      cutDrive: p[5],
      cells: Uint32Array.from(at(0)),
      farRail: at(1),
      farWalk: at(2),
      farDrive: at(3),
    };
  }

  /** Every period has reached `round`: build the view the UI reads. */
  private publish(s: Solve, round: number) {
    const res = s.results as Result[];
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
      const loads: Float32Array[] = [];
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
        if (PERIODS[q].demand === "high")
          for (let k = 0; k < 2 * n; k++) fullest = Math.max(fullest, r.loadOfCrush[o + k]);
      });
      lines.set(id, { ridersPerDay: boardings.reduce((x, y) => x + y, 0), fullestPct: Math.round(100 * fullest), boardings, loads });
    }
    const pct = (v: number) => (trips > 0 ? (100 * v) / trips : 0);
    demandView.value = {
      version: s.version,
      round,
      split: { train: pct(rail), walk: pct(walk), drive: Math.max(0, 100 - pct(rail) - pct(walk)) },
      trainShareByPeriod: byPeriod,
      peakHourFactor: PERIODS.map((_, q) => this.periodTable[q * PERIOD_STRIDE + 5] ?? 1),
      tripsPerDay: trips,
      railTripsPerDay: rail,
      lines,
      stations,
    };
    if (round === 0) s.firstAt = performance.now();
    if (round >= ROUNDS) {
      s.refinedAt = performance.now();
      if (s === this.cur) demandUpdating.value = false;
    }
  }
}

let client: DemandClient | null = null;

const TEST = new URLSearchParams(location.search).get("demandTest") === "1";

/** Start the pool for a city. With `?demandTest=1` it solves T-005's hand-made network. */
export function startDemand(city = "nyc") {
  if (client) return client;
  const c = (client = new DemandClient(city));
  setDemandQueries({ cityCells: () => c.cityCells(), cellModes: () => c.cellModes(), stationRiders: (id, top) => c.stationRiders(id, top), flows: (end, cells) => c.flows(end, cells) });
  if (TEST) c.submitNetwork(demandTestSnapshot());
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twDemand = { client: c, log: c.log, start: c.start, resubmit: () => c.resubmit(), view: () => demandView.peek(), cellModes: () => c.cellModes(), stationRiders: (id: string) => c.stationRiders(id), cityCells: () => c.cityCells(), flows: (end: "home" | "work", cells: Uint32Array) => c.flows(end, cells) };
  return c;
}

/** Hand the clock worker's NetworkSnapshot to the pool (from clockClient's onNetworkSnapshot). */
export function submitNetwork(snap: NetworkSnapshot) {
  if (!TEST) client?.submitNetwork(snap);
}
