// T-4220 stage 1 (notes/T-4220.md): the figures on screen as the solving workers hold them. Before a whole solve
// without the barrier sets anything up, each solving worker sends what it holds of every population whose figures are
// the ones shown (`shown`); the page keeps it, merged by population and period, until a final replaces what is shown.
// Later stages solve an edit made during that whole solve aside, in the focus pool, against it.

import type { CityId } from "../game/cities";
import { PERIODS } from "../game/periods";
import type { Box, PopView } from "./demandClusters";
import { sameView } from "./demandView";

/** A snapshot is not sent past this (the design's first guess; Asia's was estimated at 60-90 MB) */
export const SNAP_MAX_BYTES = 120 * 2 ** 20;
/** T-4220 stage 3: a side job solves at most this many fresh zones (T-2188's Asia train edits touched 1-5; first guess) */
export const ASIDE_MAX_ZONES = 5;

/** A network's fare settings the market prices by (`FareOf` in demand.worker.ts): the fares, each line's own multiples
 * and distance fare, each station's fare zone, by id */
export interface SnapFares { fare: string; lines: Map<string, string>; stations: Map<string, number> }

/** A network's line and station ids and what its route nodes are (`NetIds` in demand.worker.ts): route nodes on it
 * are carried to another network's by line and station ids (`nodeMap`) */
export interface SnapIds { lines: string[]; stations: string[]; lineN: Uint32Array; stops: Uint32Array; tph: Float32Array; times: Float32Array }

/** What a population's shown figures were solved on, shared by its periods in one snapshot */
export interface SnapOn {
  /** the zones of the cluster it was solved in, and its market key ("" none) */
  zones: CityId[];
  marketKey: string;
  /** a city population's view (`cityOn`): its network within the cluster, and the maps from it */
  view?: PopView;
  /** the market's route nodes on network `ids` (`marketOn`; 0xffffffff where gone) */
  nodes?: Uint32Array;
  ids?: SnapIds;
  /** the fare settings of the network it last solved on as the main pool tests an edit's against (`marketFare`; one
   * object for every population solved in one job) */
  fares?: SnapFares;
}

/** One period of a population as shown, every array on its own route nodes (its view's) */
export interface SnapPeriod {
  on: SnapOn;
  /** `heldQ`: the period signature its figures are for ("" none; a local solve's ends in "~local") */
  psig: string;
  /** `trail`: per round, the loads the round after it was priced on (seg, board and alight, 3 x route nodes) */
  trail: (Float32Array | undefined)[];
  /** `boardsOf`: the free-flow pass's boardings, and the last round's before the cap */
  free: Float32Array | null;
  rounds: Float32Array | null;
  seg: Float32Array;
  board: Float32Array;
  alight: Float32Array;
  loadOfCrush: Float32Array;
  fareX: Float32Array;
  baseX: Float32Array;
  xferIn: Float32Array;
  summary: Float64Array;
  /** an everyday layer's `kind_totals` and `part_loads` */
  kinds?: Float64Array;
  partLoads?: Float32Array;
  /** the market's `commute_loads`, `everyday_loads` and `lean_loads` */
  commute?: Float32Array;
  everyday?: Float32Array;
  lean?: Float32Array;
  /** `passByRound`: ms of its pass in each round of the period's last whole solve here */
  pass: number[];
  /** T-1735: the market's window shown with it, the local result's new loads of the window's pairs less their old
   * (`marketBlocks`' blocks: every pair's, the commutes', each everyday purpose's), on network `ids`'s route nodes */
  delta?: { ids: SnapIds; blocks: Float32Array[] };
}

export interface SnapPop {
  kind: "commute" | "everyday" | "market";
  /** a city population's zone, a market part's key */
  key: string;
  /** a city population's fare multiples (`popScales`) */
  scales?: { base: number; rate: number };
  /** an everyday layer's purposes (`purpose_count`) */
  purposes?: number;
  /** by period, those the worker holds */
  periods: (SnapPeriod | null)[];
}

export interface Snapshot {
  pops: SnapPop[];
  /** the sending worker's periods */
  periods: number[];
  /** a part of the job's populations, merged but not counted as the worker's snapshot: a cluster's sent again once its
   * result is shown (a solve of several clusters), or a local stage's touched ones as they were before it set them up,
   * when its result is never shown (`whole` came first, several solving workers: stage 8); else the job's all */
  again?: boolean;
  /** the job's zones with their cells' box and level, the market's 6 km zones and the clustering flags, for the
   * newest network's clusters to be made as the main pool makes them (absent with `again`) */
  zones?: { id: CityId; box: Box; level: number; day: boolean }[];
  marketZones?: Float64Array | null;
  flags?: { whole: boolean; oneMarket: boolean; wide: boolean };
  /** its arrays' bytes, and the ms to copy them */
  bytes: number;
  ms: number;
  /** past `SNAP_MAX_BYTES` (`bytes` where it stopped): nothing sent */
  over?: boolean;
}

/** a solving worker's whole snapshot as the store took it: its ms and bytes, and the bytes of views and network ids it
 * sent that the store already held (T-4220 stage 8) */
export interface Sent { ms: number; bytes: number; shared: number }

/** T-4220 stage 3: what the page holds of the figures on screen (`ShownStore`), for a side job in the focus pool */
export interface AsideShown {
  /** the store's `gen` it was taken at */
  gen: number;
  pops: SnapPop[];
  zones: NonNullable<Snapshot["zones"]>;
  marketZones: Float64Array | null;
  flags: NonNullable<Snapshot["flags"]>;
}

/** T-4220 stage 3: why no side job can be planned (today's wait stays): no snapshot held, the zones open since it are
 * not its own, the fares changed, the edit joins or splits a cluster it holds, a population it needs is missing or
 * not whole, a frozen one has no full trail of a period to solve, or too many zones are fresh. Stage 4: an intercity
 * edit (the main pool's local stage would open a market window, T-1735, which a side job keeps frozen), a fresh zone not
 * open in the focus worker at the level the snapshot holds it, or (the page's) held at another level by the main pool
 * now */
export type AsideRefusal = "snapshot" | "zones" | "fares" | "clusters" | "partial" | "trail" | "many" | "intercity" | "open" | "level";

/** T-4220 stage 3: the side job's plan for a network against the figures on screen */
export interface AsideAnswer {
  /** absent: planned (with no fresh zone, the edit moves nothing past `sameView`'s tolerance) */
  refused?: AsideRefusal;
  /** what was refused, for the dev log */
  why?: string;
  /** the zones a population of whose view on the network is not its shown one's, with the level the snapshot's solve held
   * them at, in the registry's order */
  fresh: { id: CityId; level: number }[];
  /** each cluster with a fresh zone: its zones, its fresh ones, the periods (bits) every population keeps as shown, those
   * solved, and the expected ms of their solve from the snapshot's passes (null: not known) */
  clusters: { zones: CityId[]; fresh: CityId[]; keep: number; solve: number[]; ms: number | null }[];
  /** the periods any cluster solves */
  periods: number[];
  /** the clusters' expected ms added up (null: one is not known), without the pool's start and the zones' opening */
  ms: number | null;
  /** planning took, ms */
  planMs: number;
}

const NO32 = new Float32Array(0), NO64 = new Float64Array(0);
function sameArr(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false;
  return true;
}
/** the same values to the bit, NaN where the other has NaN */
function sameBits(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k] && (a[k] === a[k] || b[k] === b[k])) return false;
  return true;
}
/** T-4220 stage 8: two copies of one view, as two solving workers send theirs of the same solve */
function sameViewBits(a: PopView, b: PopView): boolean {
  if (a === b) return true;
  if (!sameBits(a.stLocal, b.stLocal) || !sameBits(a.nodeGlobal, b.nodeGlobal) || !sameBits(a.pads, b.pads)) return false;
  const x = a.net as unknown as Record<string, unknown>, y = b.net as unknown as Record<string, unknown>, keys = Object.keys(x);
  return keys.length === Object.keys(y).length && keys.every((k) => {
    const u = x[k], v = y[k];
    return ArrayBuffer.isView(u) && ArrayBuffer.isView(v) ? sameBits(u as unknown as ArrayLike<number>, v as unknown as ArrayLike<number>) : JSON.stringify(u) === JSON.stringify(v);
  });
}
const sameIdsBits = (a: SnapIds, b: SnapIds) => a === b || (sameBits(a.lineN, b.lineN) && sameBits(a.stops, b.stops) && sameBits(a.tph, b.tph) && sameBits(a.times, b.times) && a.lines.join("\n") === b.lines.join("\n") && a.stations.join("\n") === b.stations.join("\n"));
/** Two periods' figures solved on the same route nodes: in one cluster, a city's on views of the same inputs (`sameView`), the
 * market's on the same nodes of one network */
function sameOn(a: SnapOn, b: SnapOn): boolean {
  if (a === b) return true;
  if (a.marketKey !== b.marketKey || a.zones.length !== b.zones.length || a.zones.some((id, k) => id !== b.zones[k])) return false;
  if (a.view || b.view) return !!a.view && !!b.view && sameView(a.view, b.view);
  if (!a.nodes || !b.nodes || !a.ids || !b.ids) return false;
  const x = a.ids, y = b.ids;
  return sameArr(a.nodes, b.nodes) && sameArr(x.lineN, y.lineN) && sameArr(x.stops, y.stops) && x.lines.join("\n") === y.lines.join("\n") && x.stations.join("\n") === y.stations.join("\n");
}

/**
 * T-4220 stage 2: a population as a snapshot holds it, standing in for it in a cluster's figures: it answers the calls that
 * read a population (`seg` ... `lean_loads`, `solved_mask`, `purpose_count`) as the WASM population did when the snapshot
 * was taken, a period it does not hold with empty arrays as they do. It returns the snapshot's own arrays: read them, never
 * write or transfer them. Never set up or solved.
 */
export class ShownPop {
  readonly kind: SnapPop["kind"];
  /** a city population's zone, a market part's key */
  readonly key: string;
  /** what its periods' figures were solved on, when they all agree (`sameOn`); else null: it cannot stand in */
  readonly on: SnapOn | null;
  constructor(readonly pop: SnapPop) {
    this.kind = pop.kind;
    this.key = pop.key;
    const ons = pop.periods.flatMap((at) => (at ? [at.on] : []));
    this.on = ons.length && ons.every((o) => sameOn(ons[0], o)) ? ons[0] : null;
  }
  private at(q: number): SnapPeriod | null { return this.pop.periods[q] ?? null; }
  solved_mask(): number { return this.pop.periods.reduce((m, at, q) => (at ? m | (1 << q) : m), 0); }
  /** `heldQ`'s period signature ("" none) */
  psig(q: number): string { return this.at(q)?.psig ?? ""; }
  purpose_count(): number { return this.pop.purposes ?? 0; }
  seg(q: number): Float32Array { return this.at(q)?.seg ?? NO32; }
  board(q: number): Float32Array { return this.at(q)?.board ?? NO32; }
  alight(q: number): Float32Array { return this.at(q)?.alight ?? NO32; }
  load_of_crush(q: number): Float32Array { return this.at(q)?.loadOfCrush ?? NO32; }
  fare_extra(q: number): Float32Array { return this.at(q)?.fareX ?? NO32; }
  base_extra(q: number): Float32Array { return this.at(q)?.baseX ?? NO32; }
  transfers_in(q: number): Float32Array { return this.at(q)?.xferIn ?? NO32; }
  summary(q: number): Float64Array { return this.at(q)?.summary ?? NO64; }
  kind_totals(q: number): Float64Array { return this.at(q)?.kinds ?? NO64; }
  part_loads(q: number): Float32Array { return this.at(q)?.partLoads ?? NO32; }
  commute_loads(q: number): Float32Array { return this.at(q)?.commute ?? NO32; }
  everyday_loads(q: number): Float32Array { return this.at(q)?.everyday ?? NO32; }
  lean_loads(q: number): Float32Array { return this.at(q)?.lean ?? NO32; }
  /** a market's T-1735 window shown with it in period q (`SnapPeriod.delta`), on the network it was solved on */
  delta(q: number): SnapPeriod["delta"] { return this.at(q)?.delta; }
}

/** The typed arrays' bytes of a view (its network's and its maps) */
export function viewBytes(v: PopView): number {
  let n = v.stLocal.byteLength + v.nodeGlobal.byteLength + v.pads.byteLength;
  for (const x of Object.values(v.net)) if (ArrayBuffer.isView(x)) n += x.byteLength;
  return n;
}
const idsBytes = (ids: SnapIds) => ids.lineN.byteLength + ids.stops.byteLength + ids.tph.byteLength + ids.times.byteLength;

/** The typed arrays' bytes of a period, its views and ids counted once over `seen` */
function periodBytes(at: SnapPeriod, seen: Set<object>): number {
  let n = 0;
  for (const x of Object.values(at)) if (ArrayBuffer.isView(x)) n += x.byteLength;
  for (const t of at.trail) n += t?.byteLength ?? 0;
  for (const b of at.delta?.blocks ?? []) n += b.byteLength;
  const once = (o: object | undefined, bytes: () => number) => { if (o && !seen.has(o)) { seen.add(o); n += bytes(); } };
  once(at.on.view, () => viewBytes(at.on.view!));
  once(at.on.ids, () => idsBytes(at.on.ids!));
  once(at.delta?.ids, () => idsBytes(at.delta!.ids));
  return n;
}

/**
 * The page's store of the workers' snapshots: per population (`kind:key`) its periods, each with the version of the
 * solve whose snapshot sent it. A solve cut short and started again (T-3925) sends only what it still holds as shown,
 * so the rest stays as the snapshot before had it; a final shown drops every period sent before it. A period is never
 * replaced by one from an older solve, and nothing older than the final on screen is taken.
 */
export class ShownStore {
  readonly pops = new Map<string, { pop: SnapPop; from: number[] }>();
  /** bumped whenever what is held changes: a focus worker holding this generation's copy needs no other */
  gen = 0;
  /** the newest whole snapshot's zones, market zones and flags */
  zones: Snapshot["zones"] = [];
  marketZones: Float64Array | null = null;
  flags: Snapshot["flags"] = { whole: false, oneMarket: false, wide: false };
  /** per version, each solving worker's whole snapshot's ms and bytes, as it came, and the bytes of its views and network
   * ids not kept since the store held the same (`shared`) */
  private got = new Map<number, Map<number, Sent>>();
  /** no snapshot of this version or older is taken: a final of it is shown, or the store was emptied after it */
  private floor = -1;
  /** from this version on, what is on screen is not all known (a worker's snapshot or the merged store past the cap):
   * nothing is taken until a final of it or later is shown, since a solve started again sends only what it still holds */
  private lostFrom: number | null = null;
  /** the version `zones`, `marketZones` and `flags` are from */
  private metaAt = -1;

  /** worker `i`'s snapshot for solve `version`: "over" when it or the merged store passed the cap (everything held
   * dropped), "ignored" when it is older than what is on screen or what is on screen is not all known */
  merge(version: number, i: number, snap: Snapshot): "held" | "over" | "ignored" {
    const sent: Sent = { ms: snap.ms, bytes: snap.bytes, shared: 0 };
    if (!snap.again) {
      const g = this.got.get(version) ?? new Map<number, Sent>();
      g.set(i, sent);
      this.got.set(version, g);
      for (const v of this.got.keys()) if (v < version - 8) this.got.delete(v);
    }
    if (version <= this.floor) return "ignored";
    // (what is on screen is then not all known: a side job would stand on a hole)
    if (snap.over) return this.lose(version);
    if (this.lostFrom !== null && version >= this.lostFrom) return "ignored";
    sent.shared = this.share(snap);
    let changed = false;
    for (const p of snap.pops) {
      const key = `${p.kind}:${p.key}`, had = this.pops.get(key);
      if (!had) { this.pops.set(key, { pop: p, from: p.periods.map((x) => (x ? version : -1)) }); changed = true; continue; }
      const take = PERIODS.map((_, q) => !!p.periods[q] && had.from[q] <= version);
      if (!take.some(Boolean)) continue;
      had.pop = { ...p, periods: PERIODS.map((_, q) => (take[q] ? p.periods[q] : had.pop.periods[q]) ?? null) };
      take.forEach((t, q) => { if (t) had.from[q] = version; });
      changed = true;
    }
    if (!snap.again && version >= this.metaAt) {
      this.metaAt = version;
      const zones = snap.zones ?? [], mz = snap.marketZones ?? null, flags = snap.flags ?? this.flags;
      // (a solve started again after T-3925's cut sends the same zones and flags: no new generation, so the side pool's
      // copy stands and a side job planned on it goes on)
      if (JSON.stringify(zones) !== JSON.stringify(this.zones) || !(mz === this.marketZones || (!!mz && !!this.marketZones && sameArr(mz, this.marketZones))) || JSON.stringify(flags) !== JSON.stringify(this.flags)) changed = true;
      this.zones = zones;
      this.marketZones = mz;
      this.flags = flags;
    }
    if (changed) this.gen++;
    // (each worker's is capped, the page holds them all)
    return this.bytes() > SNAP_MAX_BYTES ? this.lose(version) : "held";
  }

  /**
   * T-4220 stage 8: each solving worker sends its own copy of a population's view and of the market's route nodes and
   * network ids, though it holds only its own periods. Where the store already holds the same to the bit for that
   * population, the snapshot's periods take the one held, so the page, and the side pool that gets a copy of the store,
   * hold one per zone however many workers solved its periods (up to 5, Settings' Cores). Returns the bytes not kept.
   */
  private share(snap: Snapshot): number {
    const canon = new Map<object, object>();
    let saved = 0;
    const one = <T extends object>(x: T, held: (T | undefined)[], same: (a: T, b: T) => boolean, size: (x: T) => number): T => {
      let c = canon.get(x) as T | undefined;
      if (!c) {
        c = held.find((h): h is T => !!h && same(h, x)) ?? x;
        canon.set(x, c);
        if (c !== x) saved += size(x);
      }
      return c;
    };
    for (const p of snap.pops) {
      const had = this.pops.get(`${p.kind}:${p.key}`)?.pop.periods.filter((at): at is SnapPeriod => !!at);
      if (!had?.length) continue;
      for (const at of p.periods) {
        if (!at) continue;
        const on = at.on;
        if (on.view) on.view = one(on.view, had.map((h) => h.on.view), sameViewBits, viewBytes);
        if (on.nodes) on.nodes = one(on.nodes, had.map((h) => h.on.nodes), sameBits, (n) => n.byteLength);
        if (on.ids) on.ids = one(on.ids, had.map((h) => h.on.ids), sameIdsBits, idsBytes);
        if (at.delta) at.delta.ids = one(at.delta.ids, had.map((h) => h.delta?.ids), sameIdsBits, idsBytes);
      }
    }
    return saved;
  }

  private lose(version: number): "over" {
    this.empty();
    this.lostFrom = Math.min(this.lostFrom ?? version, version);
    return "over";
  }

  /** a final of `version` is shown: every period sent with it or before goes */
  dropTo(version: number) {
    this.floor = Math.max(this.floor, version);
    if (this.lostFrom !== null && version >= this.lostFrom) this.lostFrom = null;
    let dropped = false;
    for (const [key, e] of this.pops) {
      e.from.forEach((v, q) => { if (v >= 0 && v <= version) { e.from[q] = -1; e.pop.periods[q] = null; dropped = true; } });
      if (e.from.every((v) => v < 0)) this.pops.delete(key);
    }
    if (dropped) this.gen++;
    if (!this.pops.size) this.empty();
  }

  /** everything goes (another game, or demand stopped), and no snapshot of solve `upTo` or older is taken after */
  clear(upTo: number) {
    this.empty();
    this.floor = Math.max(this.floor, upTo);
    this.lostFrom = null;
  }

  private empty() {
    if (this.pops.size || this.zones?.length || this.marketZones) this.gen++;
    this.pops.clear();
    this.zones = [];
    this.marketZones = null;
  }

  /** the typed arrays held, views and ids counted once */
  bytes(): number {
    const seen = new Set<object>();
    let n = this.marketZones?.byteLength ?? 0;
    for (const { pop } of this.pops.values()) for (const at of pop.periods) if (at) n += periodBytes(at, seen);
    return n;
  }

  /** the workers' whole snapshots of `version` so far (worker -> ms and bytes) */
  sent(version: number): ReadonlyMap<number, Sent> {
    return this.got.get(version) ?? new Map();
  }

  /** T-4220 stage 3: what a side job is planned against, or null unless every population held has every period. Its arrays are the store's own:
   * posted (copied), never transferred, since later snapshots merge into them */
  forAside(): AsideShown | null {
    if (!this.pops.size || !this.zones?.length) return null;
    // (every period of every population: each solving worker sends its own, T-4277)
    if (![...this.pops.values()].every((e) => e.pop.periods.every(Boolean))) return null;
    return { gen: this.gen, pops: [...this.pops.values()].map((e) => e.pop), zones: this.zones, marketZones: this.marketZones, flags: this.flags ?? { whole: false, oneMarket: false, wide: false } };
  }

  /** the periods any population is held for */
  periods(): number {
    const qs = new Set<number>();
    for (const { pop } of this.pops.values()) pop.periods.forEach((x, q) => { if (x) qs.add(q); });
    return qs.size;
  }
}
