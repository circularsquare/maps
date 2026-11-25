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
  /** a cluster's populations sent again once its result is shown (a solve of several clusters); else the job's all */
  again?: boolean;
  /** the job's zones with their cells' box and level, the market's 6 km zones and the clustering flags, for the
   * newest network's clusters to be made as the main pool makes them (absent with `again`) */
  zones?: { id: CityId; box: Box; level: number }[];
  marketZones?: Float64Array | null;
  flags?: { whole: boolean; oneMarket: boolean; wide: boolean };
  /** its arrays' bytes, and the ms to copy them */
  bytes: number;
  ms: number;
  /** past `SNAP_MAX_BYTES` (`bytes` where it stopped): nothing sent */
  over?: boolean;
}

const NO32 = new Float32Array(0), NO64 = new Float64Array(0);
function sameArr(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false;
  return true;
}
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
  /** the newest whole snapshot's zones, market zones and flags */
  zones: Snapshot["zones"] = [];
  marketZones: Float64Array | null = null;
  flags: Snapshot["flags"] = { whole: false, oneMarket: false, wide: false };
  /** per version, each solving worker's whole snapshot's ms and bytes, as it came */
  private got = new Map<number, Map<number, { ms: number; bytes: number }>>();
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
    if (!snap.again) {
      const g = this.got.get(version) ?? new Map<number, { ms: number; bytes: number }>();
      g.set(i, { ms: snap.ms, bytes: snap.bytes });
      this.got.set(version, g);
      for (const v of this.got.keys()) if (v < version - 8) this.got.delete(v);
    }
    if (version <= this.floor) return "ignored";
    // (what is on screen is then not all known: a side job would stand on a hole)
    if (snap.over) return this.lose(version);
    if (this.lostFrom !== null && version >= this.lostFrom) return "ignored";
    for (const p of snap.pops) {
      const key = `${p.kind}:${p.key}`, had = this.pops.get(key);
      if (!had) { this.pops.set(key, { pop: p, from: p.periods.map((x) => (x ? version : -1)) }); continue; }
      const take = PERIODS.map((_, q) => !!p.periods[q] && had.from[q] <= version);
      if (!take.some(Boolean)) continue;
      had.pop = { ...p, periods: PERIODS.map((_, q) => (take[q] ? p.periods[q] : had.pop.periods[q]) ?? null) };
      take.forEach((t, q) => { if (t) had.from[q] = version; });
    }
    if (!snap.again && version >= this.metaAt) {
      this.metaAt = version;
      this.zones = snap.zones ?? [];
      this.marketZones = snap.marketZones ?? null;
      this.flags = snap.flags ?? this.flags;
    }
    // (each worker's is capped, the page holds them all)
    return this.bytes() > SNAP_MAX_BYTES ? this.lose(version) : "held";
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
    for (const [key, e] of this.pops) {
      e.from.forEach((v, q) => { if (v >= 0 && v <= version) { e.from[q] = -1; e.pop.periods[q] = null; } });
      if (e.from.every((v) => v < 0)) this.pops.delete(key);
    }
    if (!this.pops.size) this.empty();
  }

  /** everything goes (another game, or demand stopped), and no snapshot of solve `upTo` or older is taken after */
  clear(upTo: number) {
    this.empty();
    this.floor = Math.max(this.floor, upTo);
    this.lostFrom = null;
  }

  private empty() {
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
  sent(version: number): ReadonlyMap<number, { ms: number; bytes: number }> {
    return this.got.get(version) ?? new Map();
  }

  /** the periods any population is held for */
  periods(): number {
    const qs = new Set<number>();
    for (const { pop } of this.pops.values()) pop.periods.forEach((x, q) => { if (x) qs.add(q); });
    return qs.size;
  }
}
