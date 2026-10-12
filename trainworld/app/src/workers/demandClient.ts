// Main-thread side of the shared-world demand pool (T-116/T-117).
//
// - Fetches both local city packs plus the regional trip market once. Each worker owns the
//   three populations; the pool budget applies to the whole world, not each city.
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
import { activeCity, CITIES, getWorldFrame, HOME_CITY, LEGACY_FRAME, MARKET_PACK, marketToWorld, worldFrameEpoch, type CityId } from "../game/cities";
import type { FrameSpec } from "../game/frame";
import {
  cityDemandViews, cityBaseSplits, intercityDemand,
  baseSplit,
  demandStep,
  demandUpdating,
  demandView,
  setDemandQueries,
  type CellModes,
  type CellRiders,
  type CityCells,
  type Flows,
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
  stations: { id: string; x: number; y: number }[];
  lines: { id: string; stops: string[]; tph: [number, number, number]; hopS: number[][][]; hopKm?: number[][]; dwellS: number[][][]; cars: number }[];
}

type Result = Extract<FromDemand, { kind: "result" }>;
type QueryKind = "cells" | "subSums" | "cellModes" | "station" | "flowRail" | "flows" | "purposes" | "icSel" | "icStation";
type Reply = Extract<FromDemand, { kind: QueryKind }>;
type QueryMessage = Extract<ToDemand, { kind: QueryKind }>;
type WithoutCity<T> = T extends unknown ? Omit<T, "city" | "round"> : never;
type Query = WithoutCity<QueryMessage>;
const ALL_PERIODS = (1 << PERIODS.length) - 1;
const frameOf = (): FrameSpec => { const f = getWorldFrame(); return { kind: f.kind, lon: f.lon, lat: f.lat }; };
/** A point the regional market reports (in its own pack frame) as a world point. */
const marketXY = (x: number, y: number): { x: number; y: number } => marketToWorld(x, y);
/** Where the zones' own x, y start in a longer-trip answer of `len` numbers: they close it, are the
 * same on every worker and so are not added up. For M markets a selection has 1+8*M header
 * numbers then 7 per zone; a station has 6*M header numbers then 5 per zone. The trailing
 * zone metadata is x, y, market and is never summed across workers. */
function intercityZoneXY(kind: "icSel" | "icStation", len: number): number {
  const head = kind === "icSel" ? 1 + 8 * INTERCITY_PLACES.length : 6 * INTERCITY_PLACES.length;
  return head + (kind === "icSel" ? ((len - head) / 7) * 4 : ((len - head) / 5) * 2);
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
  periodMs: number[][];
  setupMs: number[];
  /** WASM linear memory per worker after its last stage, bytes */
  memBytes: number[];
}

function serviceFromSnapshot(s: NetworkSnapshot): Service {
  return {
    frame: s.frame ?? LEGACY_FRAME,
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
    frame: svc.frame,
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
  /** The pool's workers that solve periods, then one more with none (`aux`, demand.worker.ts). */
  private workers: Worker[] = [];
  private computeN = 0;
  private periodsOf: number[][] = [];
  /** every worker's sums for a round of the newest solve, sent unasked (T-119): `version:round:city` */
  private pushed = new Map<string, { mask: number; sums: Float32Array; everyday: Float32Array | null }[]>();
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
  private onReady: ((ok: boolean) => void)[] = [];
  private focusEpoch = 0;
  private focusCity = activeCity.peek();
  private focusFrame = worldFrameEpoch.peek();
  private cellsP: Promise<CityCells | null> | null = null;
  private modesP: { version: number; round: number; p: Promise<CellModes | null> } | null = null;
  readonly start: { packMs?: number; postMs?: number; workers: { wasmMs: number; openMs: number; warmMs: number; memBytes: number }[] } = { workers: [] };
  status = "loading";

  constructor(_city: string) {
    effect(() => { activeCity.value; worldFrameEpoch.value; this.syncFocus(); });
    const n = (this.computeN = poolSize());
    for (let i = 0; i <= n; i++) {
      const w = new Worker(new URL("./demand.worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<FromDemand>) => this.receive(i, e.data);
      w.onerror = (e) => this.fail(`worker ${i}: ${e.message}`);
      this.workers.push(w);
      this.periodsOf.push(i < n ? PERIODS.map((_, q) => q).filter((q) => q % n === i) : []);
    }
    this.load();
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
  }

  private fail(why: string) {
    this.status = "failed: " + why;
    demandUpdating.value = false;
    this.onReady.splice(0).forEach(f => f(false));
    for (const q of this.pendingQueries.values()) q.done([]);
    this.pendingQueries.clear();
    console.error("demand:", why);
  }

  private async load() {
    const t0 = performance.now();
    const fetchOk = async (path: string) => {
      const r = await fetch(new URL(path, document.baseURI));
      if (!r.ok) throw new Error(`${r.status} ${r.url}`);
      return r;
    };
    try {
      const [packs, intercity] = await Promise.all([
        Promise.all(CITIES.map(async city => {
          const [header, bin] = await Promise.all([fetchOk(`packs/${city.id}.json`).then(r => r.text()), fetchOk(`packs/${city.id}.bin`).then(r => r.arrayBuffer())]);
          return { id: city.id, header, bin };
        })),
        fetchOk(`packs/${MARKET_PACK}.json`).then(r => r.text()),
      ]);
      this.start.packMs = performance.now() - t0;
      const t1 = performance.now();
      this.workers.forEach((w, i) => {
        const own = packs.map(p => ({ ...p, bin: i === this.workers.length - 1 ? p.bin : p.bin.slice(0) }));
        const msg: ToDemand = { kind: "init", packs: own, intercity, periods: this.periodsOf[i], aux: i === this.computeN };
        w.postMessage(msg, own.map(p => p.bin));
      });
      this.start.postMs = performance.now() - t1;
    } catch (err) { this.fail("regional packs: " + String(err)); }
  }

  private receive(i: number, msg: FromDemand) {
    if (msg.kind === "failed") return this.fail(`worker ${i}: ${msg.error}`);
    if (msg.kind === "cells" || msg.kind === "subSums" || msg.kind === "cellModes" || msg.kind === "station" || msg.kind === "flowRail" || msg.kind === "flows" || msg.kind === "purposes" || msg.kind === "icSel" || msg.kind === "icStation") {
      const w = this.pendingQueries.get(msg.id);
      if (!w) return;
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
      if (!periodsMatch(msg.periods, PERIOD_STRIDE))
        console.error("demand: game/periods.ts disagrees with sim/src/track/params.rs PERIOD_HOURS / PERIOD_LEVEL", msg.periods);
      if (!this.info && i < this.computeN) {
        this.info = msg.info;
        this.periodTable = msg.periods;
        cityBaseSplits.value = new Map(CITIES.map(({id}) => {
          const walk = 100 * msg.cityInfo[id][3]; return [id, {train: 0, walk, drive: 100 - walk}];
        }));
        const walk = 100 * msg.info[3];
        if (!demandView.peek()) baseSplit.value = { train: 0, walk, drive: 100 - walk };
      }
      if (++this.ready === this.workers.length) {
        this.status = "ready";
        this.onReady.splice(0).forEach((f) => f(true));
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
    this.pushed.clear();
    this.cur = { version, svc, layout, results: PERIODS.map(() => null), byRound: Array.from({length: ROUNDS + 1}, () => PERIODS.map(() => null)), published: -1, arrivedAt, sentAt: performance.now(), periodMs: [], setupMs: [], memBytes: [] };
    this.log.push(this.cur);
    if (this.log.length > 50) this.log.shift();
    demandUpdating.value = true;
    demandStep.value = { done: 0, of: ROUNDS + 1 };
    const msg: ToDemand = { kind: "solve", version, net, rounds: ROUNDS };
    for (const w of this.workers) w.postMessage(msg);
  }

  // --- Demand views (T-078; types and meaning in game/demand.ts) ---------------------------------

  /** Send `make(id)` to the given workers (all by default) and collect their replies. */
  private ask(make: (id: number) => Query, which: number[] = this.workers.slice(0, this.computeN).map((_, i) => i), transfer: Transferable[] = []): Promise<Reply[]> {
    this.syncFocus();
    const id = this.nextQuery++;
    const city = activeCity.peek(), epoch = this.focusEpoch;
    return new Promise((done) => {
      this.pendingQueries.set(id, { left: which.length, replies: [], done: replies => done(epoch === this.focusEpoch ? replies : []) });
      for (const i of which) this.workers[i].postMessage({...make(id), city, round: demandView.peek()?.round ?? -1}, transfer);
    });
  }

  private whenReady(): Promise<boolean> {
    if (this.status.startsWith("failed")) return Promise.resolve(false);
    if (this.ready === this.workers.length) return Promise.resolve(true);
    return new Promise((r) => this.onReady.push(r));
  }

  cityCells(): Promise<CityCells | null> {
    this.syncFocus();
    return (this.cellsP ??= this.whenReady().then(async (ok) => {
      if (!ok) return null;
      const [r] = await this.ask((id) => ({ kind: "cells", id, frame: frameOf() }), [this.computeN]);
      if (r?.kind !== "cells") return null;
      return { n: r.xy.length / 2, xy: r.xy, h3: r.h3, zone: r.zone, zoneM: r.zoneXY[0], zoneXY: r.zoneXY.subarray(1) };
    }));
  }

  cellModes(): Promise<CellModes | null> {
    this.syncFocus();
    const version = demandView.peek()?.version;
    if (version === undefined) return Promise.resolve(null);
    // cached per solve and crowding round (T-084: the dot map asks again for the refined result)
    const round = demandView.peek()?.round ?? 0;
    if (this.modesP?.version === version && this.modesP.round === round) return this.modesP.p;
    const p = (async (): Promise<CellModes | null> => {
      const city = activeCity.peek();
      let mask = 0;
      let sums: Float32Array | null = null;
      let extra: Float32Array | null = null;
      // every worker sent its sums with the round (no question to a worker busy with the next
      // round); asked for only when one is missing
      const sent = this.pushed.get(`${version}:${round}:${activeCity.peek()}`);
      const parts: { mask: number; sums: Float32Array | null; everyday: Float32Array | null }[] = sent && sent.length === this.computeN ? sent : (await this.ask((id) => ({ kind: "subSums", id, version }))).map((r) => (r.kind === "subSums" ? r : { mask: 0, sums: null, everyday: null }));
      for (const r of parts) {
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
      const [r] = await this.ask((id) => ({ kind: "cellModes", id, version, sums: sums! }), [this.computeN], [sums.buffer]);
      if (r?.kind !== "cellModes" || !r.modes) return null;
      if (activeCity.peek() !== city || demandView.peek()?.version !== version || demandView.peek()?.round !== round) return null;
      // Bubbles now count trip legs at each home/activity pair, both directions.
      // Old commuter queries count one person; convert before adding everyday travel.
      for (let k = 0; k < r.modes.length; k++) r.modes[k] = 2 * r.modes[k] + (extra?.[k] ?? 0);
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
    const city = activeCity.peek(), round = demandView.peek()?.round;
    const version = demandView.peek()?.version;
    if (version === undefined || cells.length === 0) return null;
    const e = end === "home" ? 0 : 1;
    const [parts, purposeParts] = await Promise.all([
      this.ask((id) => ({ kind: "flowRail", id, version, end: e, cells })),
      this.ask((id) => ({ kind: "purposes", id, version, end: e, cells })),
    ]);
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
    if (r?.kind !== "flows" || !r.packed) return null;
    const p = r.packed;
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
    let work: Float32Array | null = null, everyday: Float32Array | null = null;
    let purposeNames: string[] | null = null;
    mask = 0;
    for (const reply of purposeParts) {
      if (reply.kind !== "purposes" || !reply.work) return null;
      if (purposeNames && JSON.stringify(purposeNames) !== JSON.stringify(reply.names)) return null;
      purposeNames = reply.names;
      mask |= reply.mask;
      if (!work) work = reply.work.slice();
      else for (let k=0;k<work.length;k++) work[k] += reply.work[k];
      if (reply.everyday) {
        if (!everyday) everyday = reply.everyday.slice();
        else for (let k=0;k<everyday.length;k++) everyday[k] += reply.everyday[k];
      }
    }
    if (mask !== ALL_PERIODS || !work) return null;
    const purpose = (name: string, values: Float32Array) => ({name,
      outward: [0,1,2].map(m => Array.from(values.subarray(m*5,m*5+5))),
      returning: [0,1,2].map(m => Array.from(values.subarray(15+m*5,20+m*5))),
    });
    result.purposes.push(purpose("Work", work));
    if (everyday) {
      const kinds = purposeNames!.length;
      if (!kinds) return null;
      const nc = (everyday.length/kinds-30)/3, stride = 30+3*nc;
      if (!Number.isInteger(nc) || nc <= 0) return null;
      const far = [new Float32Array(nc),new Float32Array(nc),new Float32Array(nc)];
      for (let kind=0;kind<kinds;kind++) {
        const values = everyday.subarray(kind*stride,(kind+1)*stride);
        result.purposes.push(purpose(purposeNames![kind], values));
        for (let m=0;m<3;m++) for (let c=0;c<nc;c++) far[m][c] += values[30+m*nc+c];
      }
      for (let i=0;i<result.cells.length;i++) {
        const c=result.cells[i];
        far[0][c]+=result.farRail[i];far[1][c]+=result.farWalk[i];far[2][c]+=result.farDrive[i];
      }
      const used: number[]=[];
      for (let c=0;c<nc;c++) if (far[0][c]+far[1][c]+far[2][c]>0) used.push(c);
      result.cells = Uint32Array.from(used);
      result.farRail=Float32Array.from(used,c=>far[0][c]);
      result.farWalk=Float32Array.from(used,c=>far[1][c]);
      result.farDrive=Float32Array.from(used,c=>far[2][c]);
      for (let kind=0;kind<kinds;kind++) for (let m=0;m<3;m++) {
        const values=everyday.subarray(kind*stride,kind*stride+30);
        let v=0;for(let q=0;q<5;q++) v+=values[m*5+q]+values[15+m*5+q];
        if(m===0) result.rail+=v; else if(m===1) result.walk+=v;else result.drive+=v;
      }
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
    for (const r of parts) {
      if (r.kind !== kind || !r.packed) return null;
      mask |= r.mask;
      if (!sums) sums = r.packed.slice();
      else for (let k = kind === "icSel" ? 1 : 0; k < intercityZoneXY(kind, sums.length); k++) sums[k] += r.packed[k];
    }
    if (version !== demandView.peek()?.version || mask !== ALL_PERIODS) return null;
    return sums;
  }

  /** The longer trips of a bubble selection (game/demand.ts `IntercitySelection`, T-118). */
  async intercitySelection(end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null> {
    const version = demandView.peek()?.version;
    if (version === undefined || cells.length === 0) return null;
    const e = end === "home" ? 0 : 1;
    const v = await this.sumIntercity("icSel", version, (id) => ({ kind: "icSel", id, version, end: e, cells }));
    if (!v) return null;
    // per place, 8 numbers after the matched share: starts (all, rail, x, y) then ends; then per
    // far zone (all, rail) of starts and of ends, then zone x, y, market (not summed)
    const head = 1 + 8 * INTERCITY_PLACES.length, nz = (v.length - head) / 7, xy = intercityZoneXY("icSel", v.length);
    const flows = (dir: number): PlaceFlows => {
      const places: PlaceTrips[] = [];
      let all = 0, rail = 0;
      for (let p = 0; p < INTERCITY_PLACES.length; p++) {
        const o = 1 + 8 * p + 4 * dir;
        all += v[o]; rail += v[o + 1];
        if (v[o] > 0) places.push({ place: p, all: v[o], rail: v[o + 1], ...marketXY(v[o + 2] / v[o], v[o + 3] / v[o]) });
      }
      const zones: ZoneTrips[] = [];
      for (let z = 0; z < nz; z++) {
        const a = v[head + 4 * z + 2 * dir];
        if (a > 0) zones.push({ ...marketXY(v[xy + 3 * z], v[xy + 3 * z + 1]), place: v[xy + 3 * z + 2], all: a, rail: v[head + 4 * z + 2 * dir + 1] });
      }
      return { all, rail, places: places.sort((a, b) => b.all - a.all), zones: zones.sort((a, b) => b.all - a.all) };
    };
    return { version, starts: flows(0), ends: flows(1), matched: v[0] };
  }

  /** A station's longer-trip rail riders (game/demand.ts `IntercityStation`, T-118). */
  async intercityStation(stationId: string): Promise<IntercityStation | null> {
    const version = demandView.peek()?.version;
    const solve = this.log.find((s) => s.version === version);
    if (version === undefined || !solve) return null;
    const station = solve.svc.stations.findIndex((s) => s.id === stationId);
    if (station < 0) return null;
    const v = await this.sumIntercity("icStation", version, (id) => ({ kind: "icStation", id, version, station }));
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
    return { version, stationId, starts: flows(0), ends: flows(1) };
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
      let fullest = 0, intercityRidersPerDay = 0;
      const loads: Float32Array[] = [];
      res.forEach((r, q) => {
        for (let k = 0; k < n; k++) {
          // stop k is route node o + k on run 0 and o + n + (n - 1 - k) on run 1
          const a = o + k, b = o + 2 * n - 1 - k;
          boardings[k] += r.board[a] + r.board[b];
          intercityRidersPerDay += r.intercityBoard[a] + r.intercityBoard[b];
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
      lines.set(id, { intercityRidersPerDay, ridersPerDay: boardings.reduce((x, y) => x + y, 0), fullestPct: Math.round(100 * fullest), boardings, loads });
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
    const network = demandView.peek()!;
    cityDemandViews.value = new Map(CITIES.map(({id}) => {
      let trips = 0, rail = 0, walk = 0;
      const trainShareByPeriod: [string, number][] = res.map((r, q) => {
        const [, t, ra, wa] = r.citySummary[id]; trips += t; rail += ra; walk += wa;
        return [PERIODS[q].name, t > 0 ? 100 * ra / t : 0];
      });
      const pct = (n: number) => trips > 0 ? 100 * n / trips : 0;
      return [id, {...network, tripsPerDay: trips, railTripsPerDay: rail, trainShareByPeriod, split: {train: pct(rail), walk: pct(walk), drive: Math.max(0, 100 - pct(rail) - pct(walk))}}];
    }));
    const corridors = new Map<string, {from: number; to: number; trips: number; rail: number}>();
    let interTrips = 0, interRail = 0;
    for (const r of res) {
      interTrips += r.intercitySummary[1]; interRail += r.intercitySummary[2];
      for (let i = 0; i < r.corridors.length; i += 4) {
        const [from, to, trips, rail] = r.corridors.slice(i, i + 4), key = `${from}:${to}`;
        const old = corridors.get(key) ?? {from, to, trips: 0, rail: 0}; old.trips += trips; old.rail += rail; corridors.set(key, old);
      }
    }
    intercityDemand.value = {version: s.version, round, tripsPerDay: interTrips, railTripsPerDay: interRail, corridors: [...corridors.values()]};
    if (round === 0) s.firstAt = performance.now();
    if (s === this.cur) demandStep.value = { done: round + 1, of: ROUNDS + 1 };
    if (round >= ROUNDS) {
      s.refinedAt = performance.now();
      if (s === this.cur) demandUpdating.value = false;
    }
  }
}

let client: DemandClient | null = null;

const TEST = new URLSearchParams(location.search).get("demandTest") === "1";

/** Start the pool for a city. With `?demandTest=1` it solves T-005's hand-made network. */
export function startDemand(city = HOME_CITY) {
  if (client) return client;
  const c = (client = new DemandClient(city));
  setDemandQueries({ cityCells: () => c.cityCells(), cellModes: () => c.cellModes(), stationRiders: (id, top) => c.stationRiders(id, top), flows: (end, cells) => c.flows(end, cells), intercitySelection: (end, cells) => c.intercitySelection(end, cells), intercityStation: (id) => c.intercityStation(id) });
  if (TEST) c.submitNetwork(demandTestSnapshot());
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twDemand = { client: c, log: c.log, start: c.start, resubmit: () => c.resubmit(), view: () => demandView.peek(), cities: () => cityDemandViews.peek(), intercity: () => intercityDemand.peek(), activeCity, cellModes: () => c.cellModes(), stationRiders: (id: string) => c.stationRiders(id), cityCells: () => c.cityCells(), flows: (end: "home" | "work", cells: Uint32Array) => c.flows(end, cells), intercitySelection: (end: "home" | "work", cells: Uint32Array) => c.intercitySelection(end, cells), intercityStation: (id: string) => c.intercityStation(id) };
  return c;
}

/** Hand the clock worker's NetworkSnapshot to the pool (from clockClient's onNetworkSnapshot). */
export function submitNetwork(snap: NetworkSnapshot) {
  if (!TEST) client?.submitNetwork(snap);
}
