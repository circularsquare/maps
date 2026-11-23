// The rows of the clock worker's state (edges, nodes, lines) read from the track model (T-1382).
//
// Every edit used to ask the model for every edge, node (a string call for each station's name) and
// line (some twenty calls each) again, though an edit touches a few. `StateRows` keeps the rows it
// made and asks only for what could have changed:
//  - an edge's or node's numbers come from one bulk call each and are compared with the kept row's;
//    an equal row is the same object, so posting the state finds it unchanged by identity;
//  - a station's name is kept by node, and read again only for the nodes the model says had a name
//    set (`take_name_changes`);
//  - a line's heavy rows (name, stops, path, timetable, stretch speeds, passed stations, reversals)
//    are kept, and read again only for the lines the model says its service rebuilt, rescheduled or
//    recomputed (`take_changed_lines`), which is every line an edit can change them for;
//  - the phase tables of the run profiles (`phases`, 10 MB on Japan) are kept per line the same way
//    and laid end to end only when a line's differ (T-1540).
// T-4113: and an edge's or node's own calls (its kind, one-way setting, platform line, port, ...) are made only for
// the records in a region an edit dirtied (workers/regions.ts), the records an edit touched (`take_touched`), the
// ends of the track whose lines changed, and any record whose bulk numbers moved; a line's own calls only for the
// lines the model says changed, whose bulk numbers moved or that stop in those regions, and a bus or ferry line's
// row only where its route or stops reach them. A read given a focus does the regions there first and leaves the
// rest dirty for the next read, which never drops them.
// The result is the same as `new StateRows().read(...)` on the same model, rows field for field
// (checks/t1382-rows.mjs, which runs it on the Asia save too; `?patch=check` compares them in the page).
import type { TrackApi } from "../wasm/track/sim";
import type { ClockEdge, ClockLine, ClockNode } from "./protocol";
import type { OuterGroup } from "../game/types";
import type { Complexes } from "../game/complexes";
import { nameSource } from "../game/complexLead";
import { META_PER_LINE, PROFILES_PER_LINE } from "../map/profiles";
import { Regions, type Focus, type Taken } from "./regions";

/** T-4113: stations per region at most (notes/T-4113.md: the curve it was picked from) */
export const REGION_CAP = 64;
/** T-4113: past this many records dirty, a read given a focus does the focus's regions first */
export const FOCUS_BUDGET = 3000;

/** what the model answers for "every one" in `take_changed_lines` and `take_name_changes` */
const ALL = 0xffffffff;

export interface RowEnv {
  /** world metres to longitude and latitude */
  lngLat(x: number, y: number): [number, number];
  /** hours of each demand level in a day, for `line_upkeep` */
  /** hours a day in each demand period (T-3542) */
  periodHours: Float64Array;
  /** a bus line's row, and the nodes for its stops that are no station (`nodes` is the list so far) */
  busRow(a: TrackApi, id: number, nodes: ClockNode[]): NonNullable<ClockLine["bus"]>;
  /** a station's platform line, the same array while unchanged (T-1859, workers/platformLines.ts) */
  platformLine?(a: TrackApi, id: number, x: number, y: number, platform: number): Float32Array | undefined;
  /** the complexes of these station rows (T-3288), for the one name their untyped members show */
  complexes?(stations: readonly ClockNode[]): Complexes;
  /** T-4113: what else the complexes are made from (the player's links, the station buildings), as a key */
  complexKey?(a: TrackApi): string;
}

/** What a line row holds that costs several model calls and changes only when the line's service or
 * net does. `passed` is in the state, not asked for after it, so the line panel never draws new stops
 * with old gaps (T-648). */
interface Heavy {
  name: string;
  letter: string;
  group: number;
  groupName: string;
  /** T-1813: the group's bullet text and colour set by hand, "" for none */
  groupSymbol: string;
  groupColour: string;
  /** T-3423: the groups around `group`, nearest first */
  outer: OuterGroup[];
  /** the stops its trains serve and the path they run: the plan, or the part of it a partly built line runs (T-3474) */
  stops: number[];
  path: number[];
  /** T-3474: the whole plan where `stops` is only part of it, and where that part starts in it; else empty */
  planStops: number[];
  planPath: number[];
  stopStart: number;
  pathStart: number;
  /** T-3117: the return run's path where it is not `path` backward (`TrackApi.line_back_path`), else empty */
  back: number[];
  /** T-3348: the return run's stations where one differs from `stops` (`TrackApi.line_back_stops`), else empty */
  backStops: number[];
  fromStart: number[][];
  stopS: [number[], number[]];
  stretchSpeeds: number[];
  passed: number[];
  reversals: number[];
  endTurns: number[];
  ways: number[];
  oneStops: number[];
  /** T-4013: `[station, kind]` pairs (`TrackApi.line_stop_only`) */
  stopOnly: number[];
  /** T-2895: the player set its stop order by hand (`TrackApi::line_hand`) */
  hand: boolean;
}

/** a kept line: its row, and the numbers the row was made from */
interface LineKept {
  row: ClockLine;
  raw: Float64Array;
  heavy: Heavy;
  extra: number[];
  /** T-4113: the line's own calls as last read, kept while it is out of the read's scope */
  own: LineOwn;
}

/** what a line row is made of beyond the bulk numbers and its heavy parts, one call or more each */
type LineOwn = Omit<LineRead, "i" | "id" | "ok" | "running" | "bus" | "prev" | "fresh" | "heavy" | "upkeep">;

/** T-4113: a bus or ferry line's row and the stops it adds to the nodes (each once, in order), kept until its route
 * or stops are in a region a read takes, or the model says it changed */
interface BusKept {
  raw: Float64Array;
  row: NonNullable<ClockLine["bus"]>;
  stopNodes: ClockNode[];
  /** its route's and stops' points, x, y each */
  pts: Float64Array;
}

/** T-4113: what a read takes again, from `begin` */
interface ReadScope {
  all: boolean;
  /** every record visited (its bulk numbers compared); else only those in the scope, the rest kept as they are */
  walk: boolean;
  /** every line's own calls and heavy parts again */
  linesAll: boolean;
  ei: Float64Array;
  ni: Float64Array;
  li: Float64Array;
  /** edges and nodes whose own calls are made again */
  edges: Set<number>;
  nodes: Set<number>;
  /** lines whose own calls are made again, and those read again whole (their heavy parts, read already) */
  lines: Set<number>;
  fresh: Set<number>;
  heavies: Map<number, Heavy>;
  bus: Map<number, boolean>;
  taken: Taken;
  /** the model's changed lines (`take_changed_lines`) */
  dirty: Set<number>;
}

/** a line as read before its costs */
interface LineRead {
  i: number;
  id: number;
  ok: boolean;
  running: boolean;
  bus: boolean;
  prev: LineKept | undefined;
  fresh: boolean;
  heavy: Heavy;
  auto: Float64Array;
  fast: boolean;
  tram: boolean;
  loopKind: number;
  oneWay: boolean;
  oneWayByTrack: boolean;
  limit: number;
  fareDist: boolean;
  weekend: number[];
  restTph: number[];
  shift: number;
  upkeep: number;
  busRow: ClockLine["bus"];
  /** T-3333: `line_pairs`, `line_twin_of` */
  pairs: number;
  twinOf: number;
  /** T-3542 */
  tphP: number[];
  periods: number;
  periodsOwn: number;
  hours: number[];
  hoursOwn: boolean;
  rows: ClockLine["rows"];
}

/** T-3542: a line's trains an hour per period and its rows; for four or five rows, its figures per row. */
function periodsOf(a: TrackApi, id: number): Pick<LineRead, "tphP" | "periods" | "periodsOwn" | "hours" | "hoursOwn" | "rows"> {
  const periods = a.line_period_count(id), periodsOwn = a.line_periods_own(id), tphP = Array.from(a.line_tph_by_period(id));
  const hours = Array.from(a.line_hours(id)), hoursOwn = a.line_hours_own(id).length > 0;
  if (periods <= 3) return { tphP, periods, periodsOwn, hours, hoursOwn, rows: undefined };
  const auto = a.line_row_headways(id);
  return { tphP, periods, periodsOwn, hours, hoursOwn, rows: { figs: Array.from(a.line_rows(id)), set: Array.from(a.line_row_trains(id)), ...(auto.length ? { auto: Array.from(auto) } : {}), weekend: Array.from(a.line_row_weekend(id)) } };
}

export function upgradesOf(a: TrackApi): Map<number, number> {
  const u = a.station_upgrades(), out = new Map<number, number>();
  for (let k = 0; k + 1 < u.length; k += 2) out.set(u[k], u[k + 1]);
  return out;
}

const hex = (c: number) => "#" + c.toString(16).padStart(6, "0");

/** `TrackApi.line_auto`: a headway in seconds per level, or null for a line with counts by hand */
function autoOf(v: ArrayLike<number>): [number, number, number] | null {
  return v.length === 3 ? [v[0], v[1], v[2]] : null;
}

/** T-3521: a line's weekends (`TrackApi.line_weekend`), empty where they are the weekdays' at 100%. */
function weekendOf(a: TrackApi, id: number): number[] {
  const w = a.line_weekend(id);
  return w.length === 8 && (w[0] !== 100 || w[1] !== 0) ? [...w] : [];
}

function sameNumbers(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}

export class StateRows {
  private owner: TrackApi | null = null;
  private frameKey: unknown = null;
  private edgeRows = new Map<number, ClockEdge>();
  private nodeRows = new Map<number, ClockNode>();
  /** T-3288: the rows shown with a complex's name, by station (`foldNames`) */
  private folded = new Map<number, { from: ClockNode; name: string; hub: string | undefined; row: ClockNode }>();
  /** T-3750: each station's complex's name (`nameSource`), for the station buildings nobody named */
  private complexNames = new Map<number, string>();
  private names = new Map<number, string>();
  /** T-3759: each unbound station's own name, read once per change */
  private owns = new Map<number, string>();
  /** T-3759: each station's complex's name source (`nameSource`), by station, as the last rows read it */
  private heads = new Map<number, number>();
  /** T-3759: how many stations each station's complex has, as the last rows read it */
  private sizes = new Map<number, number>();
  /** T-3759: typed bound stations whose stored name is not their complex's, as the last rows read them
   * (an older save's, `takeOldNames`) */
  private apart: number[] = [];
  private lineKept = new Map<number, LineKept>();
  /** each line's phase tables (`line_phases`), by id, and all of them laid end to end as `phases`
   * answers (T-1540): the same array until a line's tables change */
  private phaseTables = new Map<number, Float32Array>();
  private phaseAll: Float32Array | null = null;
  /** the tables changed since `phaseAll` was laid out, which of them, and whether any changed in length */
  private phaseStale = false;
  private phaseDirty = new Set<number>();
  private phaseLayout = false;
  /** lines whose tables the last `lines` found may have changed, and the lines it read */
  private phasePending = new Set<number>();
  private phaseSeen = new Set<number>();
  /** how many lines' tables the last `lines` read */
  phaseReads = 0;
  /** past this many lines to read the tables of, one call for all of them is quicker (T-1540; the
   * rows check sets it to 0 to cover that way as well) */
  static wholePhases = 48;
  /** T-4113: false makes every read call every record's own calls (the walk before regions; `?bounded=0`) */
  static bounded = true;
  /** T-4113: the regions' station cap and the focus budget (the rows check sweeps them) */
  static regionCap = REGION_CAP;
  static focusBudget = FOCUS_BUDGET;
  readonly regions = new Regions(StateRows.regionCap);
  private scope: ReadScope | null = null;
  private busKept = new Map<number, BusKept>();
  /** T-4113: each node's own calls as last made */
  private nodeOwn = new Map<number, { port: boolean; brtStation: boolean; kind: number; oneHeading: number | undefined; line: Float32Array | undefined }>();
  /** T-4113: the ends of track whose lines changed in the last read, read again in this one too (a platform's
   * ways are ordered by the lines on them as the read before left them, `PlatformLines.setLines`) */
  private carry = new Set<number>();
  /** T-4113: what the last read read again, for the notes' work counts */
  work = { edges: 0, nodes: 0, lines: 0, buses: 0, leaves: 0, left: 0, busWhy: "" };

  /** Forget every row: the next read is a full one. */
  reset() {
    this.owner = null;
    this.edgeRows.clear();
    this.nodeRows.clear();
    this.folded.clear();
    this.complexNames.clear();
    this.names.clear();
    this.owns.clear();
    this.heads.clear();
    this.sizes.clear();
    this.apart = [];
    this.lineKept.clear();
    this.phaseTables.clear();
    this.phaseAll = null;
    this.phaseStale = this.phaseLayout = false;
    this.phaseDirty.clear();
    this.phasePending.clear();
    this.phaseSeen = new Set();
    this.regions.reset();
    this.scope = null;
    this.busKept.clear();
    this.nodeOwn.clear();
    this.carry.clear();
    this.lastSets.clear();
    this.lastNodes = null;
    this.lastFoldKey = "";
    this.namesMoved = [];
  }

  /** A kept model node's row (not a bus stop's). */
  nodeRow(id: number): ClockNode | undefined {
    return this.nodeRows.get(id);
  }

  /** A different model or frame than the rows were read from: start again. */
  bind(a: TrackApi, frameKey: unknown) {
    if (this.owner !== a || this.frameKey !== frameKey) {
      this.reset();
      this.owner = a;
      this.frameKey = frameKey;
    }
  }

  /** Take the model's name changes into the kept names. Called before any name is looked up. */
  private syncNames(a: TrackApi) {
    const changed = a.take_name_changes();
    if (changed.length && changed[0] === ALL) { this.names.clear(); this.owns.clear(); this.lastNodes = null; }
    else for (const id of changed) { this.names.delete(id); this.owns.delete(id); this.namesMoved.push(id); }
  }
  /** T-4113: stations named anew since the last nodes read, read again by it */
  private namesMoved: number[] = [];

  /** T-3759: a station's own name, "" while bound: kept, read once per change. */
  private own(a: TrackApi, id: number): string {
    let n = this.owns.get(id);
    if (n === undefined) {
      n = a.station_own(id);
      this.owns.set(id, n);
    }
    return n;
  }

  /** A station's name: kept, read once per change. */
  nameOf(a: TrackApi, id: number, frameKey: unknown): string {
    this.bind(a, frameKey);
    this.syncNames(a);
    return this.name(a, id);
  }

  private name(a: TrackApi, id: number): string {
    let n = this.names.get(id);
    if (n === undefined) {
      n = a.node_name(id);
      this.names.set(id, n);
    }
    return n;
  }

  /** The state's rows, a node's name or a line's heavy parts read again only where the model says
   * they changed. The nodes list is the model's nodes in order, then a bus line's stops that are no
   * station. */
  read(a: TrackApi, env: RowEnv, frameKey: unknown, ups: Map<number, number> = upgradesOf(a), focus?: Focus): { edges: ClockEdge[]; nodes: ClockNode[]; lines: ClockLine[] } {
    this.bind(a, frameKey);
    this.begin(a, focus);
    const edges = this.edges(a), nodes = this.nodes(a, env, ups);
    return { edges, nodes, lines: this.lines(a, env, nodes) };
  }

  /** T-4113: what this read takes again (`edges`, `nodes` and `lines` read it; without it they read everything):
   * the regions an edit dirtied since the last read, or with a focus past `FOCUS_BUDGET` those in it first. */
  begin(a: TrackApi, focus?: Focus) {
    const ei = a.edges_info(), ni = a.nodes_info(), li = a.lines_info();
    a.set_line_stamps(StateRows.bounded);
    const touched = a.take_touched(), changed = a.take_changed_lines();
    // every line again after a load or repricing; every edge and node after a load, the focus's first in a rows state
    const linesAll = touched[0] === ALL || (changed.length > 0 && changed[0] === ALL) || !this.lineKept.size;
    const all = !StateRows.bounded || touched[0] === ALL || !this.edgeRows.size;
    const R = this.regions;
    // (bounded: only the records an edit touched, and any added or gone, move between regions)
    R.sync(ni, ei, all ? undefined : this.touchedIds(touched));
    const nodes = new Set<number>(this.carry), edges = new Set<number>();
    /** the edit's own records' ends: their regions are the focus's */
    const own: number[] = [], mine = new Set<number>();
    this.carry = new Set();
    if (all) R.markAll();
    else {
      const ne = touched[0];
      for (let k = 1; k <= ne; k++) { R.markEdge(touched[k]); edges.add(touched[k]); mine.add(touched[k]); }
      for (let k = 1 + ne; k < touched.length; k++) { R.markNode(touched[k]); nodes.add(touched[k]); own.push(touched[k]); }
    }
    // the lines read again whole, their heavy parts now: track whose lines changed (a one-way arrow the model works
    // out from them, the order of a platform's ways) is read again at its ends
    const dirty = new Set<number>(linesAll ? [] : changed);
    let ends: Map<number, [number, number]> | null = null;
    const endsOf = (e: number) => {
      if (!ends) { ends = new Map(); for (let i = 0; i < ei.length; i += 10) ends.set(ei[i], [ei[i + 1], ei[i + 2]]); }
      return ends.get(e);
    };
    const heavies = new Map<number, Heavy>(), bus = new Map<number, boolean>(), fresh = new Set<number>();
    const moved = (p: readonly number[] | undefined, q: readonly number[] | undefined) => {
      const was = new Set(p ?? []), now = new Set(q ?? []);
      for (const v of was) if (!now.has(v)) { edges.add(v >> 1); const e = endsOf(v >> 1); if (e) { nodes.add(e[0]); nodes.add(e[1]); this.carry.add(e[0]); this.carry.add(e[1]); } }
      for (const v of now) if (!was.has(v)) { edges.add(v >> 1); const e = endsOf(v >> 1); if (e) { nodes.add(e[0]); nodes.add(e[1]); this.carry.add(e[0]); this.carry.add(e[1]); } }
    };
    for (let i = 0; i < li.length; i += 22) {
      const id = li[i], ok = li[i + 2] === 1, running = li[i + 3] === 1;
      const isBus = a.is_bus(id) || a.is_ferry(id);
      bus.set(id, isBus);
      const prev = this.lineKept.get(id);
      if (!(linesAll || !prev || dirty.has(id) || prev.row.ok !== ok || prev.row.running !== running)) continue;
      fresh.add(id);
      const h = this.heavy(a, id, ok);
      heavies.set(id, h);
      // (a line that stops or starts counting, ok or running, moves what its track derives from the lines, such as a
      // junction's crossing moves, with no entry of its path moving: its whole track is read again)
      const flip = !!prev && (prev.row.ok !== ok || prev.row.running !== running);
      if (!isBus) for (const k of ["path", "planPath", "back"] as const) if (flip) { moved(prev.heavy[k], []); moved([], h[k]); } else moved(prev?.heavy[k], h[k]);
    }
    // a line gone: its track too
    if (this.lineKept.size) {
      const now = new Set<number>(); for (let i = 0; i < li.length; i += 22) now.add(li[i]);
      for (const [id, k] of this.lineKept) if (!now.has(id) && !k.row.bus) { moved(k.heavy.path, []); moved(k.heavy.planPath, []); moved(k.heavy.back, []); }
    }
    if (mine.size) for (let i = 0; i < ei.length; i += 10) if (mine.has(ei[i])) own.push(ei[i + 1], ei[i + 2]);
    const taken = R.take(focus ? { ...focus, nodes: [...(focus.nodes ?? []), ...own] } : undefined, StateRows.focusBudget);
    for (const n of taken.nodes) nodes.add(n);
    for (const e of taken.edges) edges.add(e);
    // lines stopping where the read reads again have their own calls made again (a twin, a pair)
    const lines = new Set<number>(fresh);
    if (!linesAll && nodes.size) for (const [id, k] of this.lineKept) if (k.row.stops.some((s) => nodes.has(s))) lines.add(id);
    // (an "everything" a focus split leaves the rest to the bulk numbers now and its regions to the next read)
    // (repricing for water or terrain moves edge prices no edit touched: every record is visited then)
    const walk = !StateRows.bounded || all || linesAll;
    this.scope = { all: taken.all || all && (!focus || !StateRows.bounded), walk, linesAll, ei, ni, li, edges, nodes, lines, heavies, fresh, bus, taken, dirty };
    this.work = { edges: 0, nodes: 0, lines: 0, buses: 0, leaves: taken.leaves, left: taken.left, busWhy: "" };
  }

  /** the read's scope, begun now when `begin` was not called (a caller reading one part) */
  private scoped(a: TrackApi): ReadScope {
    if (!this.scope) this.begin(a);
    return this.scope!;
  }

  edges(a: TrackApi): ClockEdge[] {
    const sc = this.scoped(a);
    const ei = sc.ei;
    const offEdges = new Set<number>(a.off_edges());
    const hsrEdges = new Set<number>(a.hsr_edges());
    const crossoverIds = a.crossover_edges(), crossoverPartners = a.crossover_partners();
    const crossovers = new Map<number, number>(Array.from(crossoverIds, (id, k) => [id, crossoverPartners[k] === 0xffffffff ? -1 : crossoverPartners[k]]));
    const ow = a.oneway_edges(), oneWays = new Map<number, 1 | 2>();
    for (let k = 0; k + 1 < ow.length; k += 2) oneWays.set(ow[k], ow[k + 1] as 1 | 2);
    const jn = a.join_edges(), joins = new Map<number, number>();
    for (let k = 0; k + 1 < jn.length; k += 2) joins.set(jn[k], jn[k + 1]);
    const lm = a.edge_limits(), limits = new Map<number, number>();
    for (let k = 0; k + 1 < lm.length; k += 2) limits.set(lm[k], lm[k + 1]);
    const rp = a.edge_ramps(), ramps = new Map<number, number>();
    for (let k = 0; k + 1 < rp.length; k += 2) ramps.set(rp[k], rp[k + 1]);
    const out: ClockEdge[] = [];
    const kept = this.edgeRows;
    const seen = new Set<number>();
    // T-4113: a record no edit touched whose sets are as last read is kept without a look
    const patch = !sc.walk;
    const sets = { offEdges, hsrEdges, crossovers, oneWays, joins, limits, ramps };
    for (const [k, v] of Object.entries(sets)) { if (patch) StateRows.moved(this.lastSets.get("e" + k), v, sc.edges); this.lastSets.set("e" + k, v); }
    for (let i = 0; i < ei.length; i += 10) {
      const id = ei[i];
      if (patch && !sc.edges.has(id)) {
        const r = kept.get(id);
        if (r && StateRows.edgeBulkSame(r, ei, i)) { seen.add(id); out.push(r); continue; }
      }
      const off = offEdges.has(id), hsr = hsrEdges.has(id), crossover = crossovers.has(id), partner = crossovers.get(id) ?? -1;
      const tracks = ei[i + 3] as 1 | 2, built = ei[i + 4] === 1, oneWay = oneWays.get(id) ?? 0, join = joins.get(id) ?? 0, limit = limits.get(id) ?? 0, ramp = ramps.get(id) ?? 0;
      let r = kept.get(id);
      // T-4113: the edge's own calls where the read takes it again, or anything bulk about it moved
      const deep = !r || sc.all || sc.edges.has(id) || (r.a !== ei[i + 1] || r.b !== ei[i + 2] || r.tracks !== tracks || r.built !== built || r.off !== off || r.hsr !== hsr || r.oneWay !== oneWay || r.join !== join || r.limit !== limit || r.ramp !== ramp || r.crossover !== crossover || r.partner !== partner);
      if (deep) this.work.edges++;
      const kind = !deep ? r!.kind : a.edge_kind(id) >= 2 ? a.edge_kind(id) : 0 /* T-3078: 1 is high speed, which is rail */;
      const setWay = (!deep ? r!.setWay : tracks === 1 ? a.edge_oneway(id) : 0) as 0 | 1 | 2 | 3, autoWay = (!deep ? r!.autoWay : tracks === 1 ? a.edge_infer(id) : 0) as 0 | 1 | 2;
      if (!r || r.a !== ei[i + 1] || r.b !== ei[i + 2] || r.tracks !== tracks || r.built !== built || r.off !== off || r.hsr !== hsr || r.kind !== kind || r.oneWay !== oneWay || r.join !== join || r.limit !== limit || r.ramp !== ramp || r.setWay !== setWay || r.autoWay !== autoWay || r.crossover !== crossover || r.partner !== partner
        || !Object.is(r.lengthM, ei[i + 5]) || !Object.is(r.cost, ei[i + 6]) || !Object.is(r.vminKmh, ei[i + 7]) || r.levelMin !== ei[i + 8] || r.levelMax !== ei[i + 9]) {
        r = { id, a: ei[i + 1], b: ei[i + 2], tracks, built, off, hsr, kind, oneWay, join, setWay, autoWay, crossover, partner, lengthM: ei[i + 5], cost: ei[i + 6], vminKmh: ei[i + 7], limit, ramp, levelMin: ei[i + 8], levelMax: ei[i + 9] };
        kept.set(id, r);
      }
      seen.add(id);
      out.push(r);
    }
    if (kept.size !== seen.size) for (const id of kept.keys()) if (!seen.has(id)) kept.delete(id);
    return out;
  }

  nodes(a: TrackApi, env: RowEnv, ups: Map<number, number>): ClockNode[] {
    const sc = this.scoped(a);
    this.syncNames(a);
    const ni = sc.ni;
    const offStations = new Set<number>(a.off_stations());
    const namedStations = new Set<number>(a.named_stations());
    const unbound = new Set<number>(a.unbound_stations());
    // T-3332: one-track stations, 1 on double track, 2 on single
    const ot = a.one_track_stations(), ones = new Map<number, 1 | 2>();
    for (let k = 0; k + 1 < ot.length; k += 2) ones.set(ot[k], ot[k + 1] as 1 | 2);
    const onePlatforms = new Set<number>(a.one_platform_stations());
    // T-3740: the stations' fare zones (none in most games)
    const fz = a.station_fares(), fares = new Map<number, number>();
    for (let k = 0; k + 1 < fz.length; k += 2) fares.set(fz[k], fz[k + 1]);
    // T-3922: the upkeep of what is bought on each station, US$ a game day
    const uu = a.station_upgrade_upkeep(), upUpkeep = new Map<number, number>();
    for (let k = 0; k + 1 < uu.length; k += 2) upUpkeep.set(uu[k], uu[k + 1] * 1e6);
    const out: ClockNode[] = [];
    const kept = this.nodeRows;
    const seen = new Set<number>();
    const patch = !sc.walk;
    const sets = { offStations, namedStations, unbound, ones, onePlatforms, fares, upUpkeep, ups };
    for (const [k, v] of Object.entries(sets)) { if (patch) StateRows.moved(this.lastSets.get("n" + k), v, sc.nodes); this.lastSets.set("n" + k, v); }
    for (const id of this.namesMoved) sc.nodes.add(id);
    this.namesMoved = [];
    let rebuilt = !patch || !this.lastNodes;
    for (let i = 0; i < ni.length; i += 10) {
      const id = ni[i];
      if (patch && !sc.nodes.has(id)) {
        const r = kept.get(id);
        if (r && StateRows.nodeBulkSame(r, ni, i)) { seen.add(id); out.push(r); continue; }
      }
      const was0 = kept.get(id);
      const platform = ni[i + 7];
      const stationBuilt = ni[i + 8] === 1, stationOff = offStations.has(id), upgrades = ups.get(id) ?? 0, flying = (ni[i + 6] & 1) === 1, crosses = (ni[i + 6] & 2) === 2;
      const unusable = ((ni[i + 6] >> 2) & 3) as 0 | 1 | 2 | 3;
      const name = platform > 0 ? this.name(a, id) : "";
      const named = platform > 0 && namedStations.has(id);
      const own = platform > 0 && unbound.has(id) ? this.own(a, id) : "";
      const onePlatform = onePlatforms.has(id), one = ones.get(id) ?? 0;
      const fare = fares.get(id) ?? 0, upgradeUpkeep = upUpkeep.get(id) ?? 0;
      let r = kept.get(id);
      // T-4113: the node's own calls where the read takes it again, or anything bulk about it moved
      const was = this.nodeOwn.get(id);
      const deep = !r || !was || sc.all || sc.nodes.has(id) || (r.x !== ni[i + 1] || r.y !== ni[i + 2] || r.level !== ni[i + 3] || r.ports !== ni[i + 4] || r.builtPorts !== ni[i + 5] || r.flying !== flying || !!r.crosses !== crosses || (r.unusable ?? 0) !== unusable || r.platform !== platform
        || r.stationBuilt !== stationBuilt || r.stationOff !== stationOff || r.upgrades !== upgrades || r.name !== name || !!r.named !== named || (r.own ?? "") !== own || !Object.is(r.heading, ni[i + 9]) || (r.one ?? 0) !== one || !!r.onePlatform !== onePlatform || (r.fare ?? 0) !== fare || (r.upgradeUpkeep ?? 0) !== upgradeUpkeep);
      let port: boolean, brtStation: boolean, kind: number, oneHeading: number | undefined, line: Float32Array | undefined;
      if (deep) {
        this.work.nodes++;
        // T-2194: a port has no track, so no platform line
        port = platform > 0 && a.is_port(id);
        brtStation = platform > 0 && !port && a.is_brt_station(id);
        kind = a.node_kind(id) >= 2 ? a.node_kind(id) : 0;
        oneHeading = one === 1 ? a.station_one_heading(id) : undefined;
        line = platform > 0 && !port ? env.platformLine?.(a, id, ni[i + 1], ni[i + 2], platform) : undefined;
        this.nodeOwn.set(id, { port, brtStation, kind, oneHeading, line });
      } else ({ port, brtStation, kind, oneHeading, line } = was!);
      if (!r || r.x !== ni[i + 1] || r.y !== ni[i + 2] || r.level !== ni[i + 3] || r.ports !== ni[i + 4] || r.builtPorts !== ni[i + 5] || r.flying !== flying || !!r.crosses !== crosses || (r.unusable ?? 0) !== unusable || r.platform !== platform
        || r.stationBuilt !== stationBuilt || r.stationOff !== stationOff || r.upgrades !== upgrades || r.name !== name || !!r.named !== named || (r.own ?? "") !== own || !Object.is(r.heading, ni[i + 9]) || r.line !== line || !!r.port !== port || !!r.brtStation !== brtStation || (r.kind ?? 0) !== kind
        || (r.one ?? 0) !== one || r.oneHeading !== oneHeading || !!r.onePlatform !== onePlatform || (r.fare ?? 0) !== fare || (r.upgradeUpkeep ?? 0) !== upgradeUpkeep) {
        const [lng, lat] = env.lngLat(ni[i + 1], ni[i + 2]);
        const berth = port ? a.port_berth(id) : null;
        r = { id, x: ni[i + 1], y: ni[i + 2], lng, lat, level: ni[i + 3], ports: ni[i + 4], builtPorts: ni[i + 5], flying, ...(crosses ? { crosses: true as const } : {}), ...(unusable ? { unusable } : {}), platform, stationBuilt, stationOff, upgrades, ...(fare ? { fare } : {}), name, heading: ni[i + 9], ...(named ? { named: true as const } : {}), ...(own ? { own } : {}), ...(line ? { line } : {}), ...(kind ? { kind } : {}),
          ...(one ? { one } : {}), ...(oneHeading !== undefined ? { oneHeading } : {}), ...(onePlatform ? { onePlatform: true as const } : {}), ...(upgradeUpkeep ? { upgradeUpkeep } : {}),
          ...(berth && berth.length === 2 ? { port: env.lngLat(berth[0], berth[1]) } : {}), ...(brtStation ? { brtStation: true as const } : {}) };
        kept.set(id, r);
      }
      seen.add(id);
      out.push(r);
      if (r !== was0) rebuilt = true;
    }
    if (kept.size !== seen.size) { rebuilt = true; for (const id of kept.keys()) if (!seen.has(id)) kept.delete(id); }
    if (this.nodeOwn.size > seen.size) for (const id of this.nodeOwn.keys()) if (!seen.has(id)) this.nodeOwn.delete(id);
    // a station no longer there keeps no name
    if (this.names.size > seen.size) for (const id of this.names.keys()) if (!seen.has(id)) this.names.delete(id);
    // T-4113: no station row new and the links and buildings as before: the names shown are the last read's
    const key = env.complexKey?.(a) ?? "";
    if (!rebuilt && this.lastNodes && key === this.lastFoldKey) return this.lastNodes.slice();
    this.foldNames(out, env.complexes);
    this.lastNodes = out.slice();
    this.lastFoldKey = key;
    return out;
  }

  /** T-3288, T-3750, T-3759: the name each station shows (`ClockNode.name`): an unbound one its own,
   * a bound one its complex's (`nameSource`: the oldest typed name, else the oldest station's, over
   * every member, bound or not). The model keeps every station's stored name, its part in the complex's;
   * a split complex gives each part's stations theirs back. The replaced rows are kept, so an unchanged
   * one is the same object as before. */
  private foldNames(out: ClockNode[], complexesOf: RowEnv["complexes"]) {
    const stations = out.filter((r) => r.platform > 0 && !r.port);
    const cx = complexesOf?.(stations);
    const groups = new Map<number, ClockNode[]>();
    if (cx) for (const r of stations) {
      const c = cx.of.get(String(r.id));
      if (c === undefined) continue;
      const g = groups.get(c);
      if (g) g.push(r);
      else groups.set(c, [r]);
    }
    const source = new Map<number, ClockNode>();
    this.complexNames.clear();
    this.heads.clear();
    this.sizes.clear();
    this.apart = [];
    for (const [c, g] of groups) {
      const s = nameSource(g, (r) => r.id, (r) => !!r.named)!;
      // a station alone is its own complex: the complex is called what the station shows
      if (g.length === 1) {
        this.complexNames.set(s.id, s.own || s.name);
        this.heads.set(s.id, s.id);
        this.sizes.set(s.id, 1);
        continue;
      }
      source.set(c, s);
      for (const r of g) {
        this.complexNames.set(r.id, s.name);
        this.heads.set(r.id, s.id);
        this.sizes.set(r.id, g.length);
        if (r.named && !r.own && r !== s && r.name !== s.name) this.apart.push(r.id);
      }
    }
    const seen = new Set<number>();
    for (let k = 0; k < out.length; k++) {
      const r = out[k];
      if (r.platform <= 0 || r.port) continue;
      const c = cx?.of.get(String(r.id));
      const complex = c === undefined ? undefined : source.get(c)?.name;
      const name = r.own || complex || r.name;
      const hub = complex !== undefined && complex !== name ? complex : undefined;
      if (name === r.name && hub === undefined) continue;
      let f = this.folded.get(r.id);
      if (!f || f.from !== r || f.name !== name || f.hub !== hub) this.folded.set(r.id, (f = { from: r, name, hub, row: { ...r, name, ...(hub !== undefined ? { hub } : {}) } }));
      seen.add(r.id);
      out[k] = f.row;
    }
    if (this.folded.size !== seen.size) for (const id of this.folded.keys()) if (!seen.has(id)) this.folded.delete(id);
  }

  /** The name a station shows when that is not its stored one (`foldNames`), if it does. */
  shownName(id: number): string | undefined {
    return this.folded.get(id)?.name;
  }

  /** T-3759: the station whose stored name is its complex's (`nameSource`), as the last rows read it. */
  headOf(id: number): number | undefined {
    return this.heads.get(id);
  }

  /** T-3759: how many stations the complex of station `id` has, as the last rows read it (0: none known). */
  complexSize(id: number): number {
    return this.sizes.get(id) ?? 0;
  }

  /** T-3759: the typed stations that are bound but whose stored name is not their complex's, as the last
   * rows read them: an older save's, which showed their own names (`TrackApi.adopt_old_names`). */
  oldApart(): number[] {
    return this.apart;
  }

  /** The name of the complex station `id` is in (`nameSource`), as the last rows read it. */
  complexName(id: number): string | undefined {
    return this.complexNames.get(id);
  }

  /** `nodes` must have been read just before, for the bus stops that join it. A planned line's build
   * and train costs are no part of the row: a panel asks for them (`lineCost`, T-1171). */
  lines(a: TrackApi, env: RowEnv, nodes?: ClockNode[]): ClockLine[] {
    const sc = this.scoped(a);
    this.scope = null;
    const { linesAll: all, li, dirty, heavies, fresh: freshSet, bus: busOf, lines: deepLines } = sc;
    const lu = a.line_upkeep_periods(env.periodHours), upkeepOf = new Map<number, number>();
    for (let k = 0; k + 1 < lu.length; k += 2) upkeepOf.set(lu[k], lu[k + 1]);
    const kept = this.lineKept;
    const seen = new Set<number>();
    const read: LineRead[] = [];
    const refresh = new Set<number>();
    // a bus line's stops that are no station join the nodes once each, after the model's (T-1464)
    const nodeIds = new Set<number>();
    if (nodes) for (const n of nodes) nodeIds.add(n.id);
    for (let i = 0; i < li.length; i += 22) {
      const id = li[i];
      seen.add(id);
      const ok = li[i + 2] === 1, running = li[i + 3] === 1;
      const bus = busOf.get(id) ?? (a.is_bus(id) || a.is_ferry(id));
      const prev = kept.get(id);
      const raw = li.slice(i, i + 22);
      let busRow: ClockLine["bus"];
      let fresh = freshSet.has(id);
      if (bus) {
        // T-4113: a bus or ferry line's row again where its route or stops are in the regions read again
        const bk = this.busKept.get(id);
        const again = all || !StateRows.bounded || !bk || fresh || dirty.has(id) || !sameNumbers(raw, bk.raw) || this.reaches(sc.taken, bk.pts);
        if (again) {
          this.work.buses++;
          this.work.busWhy += all ? "a" : !bk ? "n" : fresh ? "f" : dirty.has(id) ? "d" : !sameNumbers(raw, bk.raw) ? "r" : "t";
          const own: ClockNode[] = [];
          busRow = env.busRow(a, id, own);
          const pts: number[] = [];
          for (const s of busRow.stops) pts.push(s.x, s.y);
          for (let k = 0; k + 1 < busRow.route.length; k += 2) pts.push(busRow.route[k], busRow.route[k + 1]);
          this.busKept.set(id, { raw, row: busRow, stopNodes: own, pts: Float64Array.from(pts) });
          if (!fresh) { fresh = true; heavies.set(id, this.heavy(a, id, ok)); }
        } else busRow = bk.row;
        for (const n of this.busKept.get(id)!.stopNodes) if (!nodeIds.has(n.id)) { nodeIds.add(n.id); nodes?.push(n); }
      }
      if (fresh) refresh.add(id);
      const heavy = fresh ? heavies.get(id) ?? this.heavy(a, id, ok) : prev!.heavy;
      const deep = all || !StateRows.bounded || fresh || !prev || deepLines.has(id) || !sameNumbers(raw, prev.raw);
      if (deep) this.work.lines++;
      const own: LineOwn = deep ? {
        auto: a.line_auto(id), fast: a.line_fast(id), tram: a.line_tram(id), loopKind: a.line_loop(id), oneWay: a.line_one_way(id), oneWayByTrack: a.line_one_way_by_track(id), limit: a.line_limit(id), fareDist: a.line_fare_dist(id), weekend: weekendOf(a, id), restTph: [...a.line_tph_rest(id)], shift: a.line_shift(id),
        ...periodsOf(a, id),
        busRow,
        pairs: bus ? 0 : a.line_pairs(id), twinOf: bus ? -1 : a.line_twin_of(id),
      } : { ...prev!.own, busRow };
      read.push({ i, id, ok, running, bus, prev, fresh, heavy, upkeep: upkeepOf.get(id) ?? 0, ...own });
    }
    if (this.busKept.size) for (const id of this.busKept.keys()) if (!seen.has(id) || !busOf.get(id)) this.busKept.delete(id);
    const out: ClockLine[] = [];
    for (const r of read) {
      const { i, id, ok, running, prev, fresh, heavy, auto, fast, tram, loopKind, oneWay, oneWayByTrack, limit, fareDist, weekend, restTph, shift, upkeep, pairs, twinOf, tphP, periods, periodsOwn, hours, hoursOwn, rows } = r;
      // the numbers behind the cheap fields: when all are as kept (and the heavy parts are), the row is the kept one
      const raw = li.slice(i, i + 22);
      const extra = [upkeep, fast ? 1 : 0, tram ? 1 : 0, loopKind, oneWay ? 1 : 0, oneWayByTrack ? 1 : 0, limit, fareDist ? 1 : 0, shift, pairs, twinOf, ...auto, weekend.length, ...weekend, ...restTph,
        ...tphP, periods, periodsOwn, ...hours, hoursOwn ? 1 : 0, ...(rows ? [rows.figs.length, ...rows.figs, rows.set.length, ...rows.set, rows.auto?.length ?? -1, ...(rows.auto ?? []), rows.weekend.length, ...rows.weekend] : [])];
      const { i: _i, id: _id, ok: _ok, running: _running, bus: _bus, prev: _prev, fresh: _fresh, heavy: _heavy, upkeep: _upkeep, ...own } = r;
      if (!fresh && prev && prev.row.bus === r.busRow && sameNumbers(raw, prev.raw) && sameNumbers(extra, prev.extra)) {
        out.push(prev.row);
        continue;
      }
      const t3 = (k: number) => [li[i + k], li[i + k + 1], li[i + k + 2]] as [number, number, number];
      const row: ClockLine = {
        upkeep,
        id, name: heavy.name, letter: heavy.letter, group: heavy.group, groupName: heavy.groupName, groupSymbol: heavy.groupSymbol, groupColour: heavy.groupColour, outer: heavy.outer, colour: hex(li[i + 1]), ok, running, broken: li[i + 4] === 1, tph: t3(5), cars: li[i + 21] || li[i + 8], carsSet: li[i + 8],
        lengthM: li[i + 9], roundTrip: t3(10), trains: t3(13), auto: autoOf(auto), delay: t3(16), dwell: li[i + 19], turnaround: li[i + 20],
        stops: heavy.stops, path: heavy.path,
        ...(heavy.planStops.length ? { planStops: heavy.planStops, planPath: heavy.planPath, stopStart: heavy.stopStart, pathStart: heavy.pathStart } : {}),
        ...(heavy.back.length ? { back: heavy.back } : {}), ...(heavy.backStops.length ? { backStops: heavy.backStops } : {}),
        fromStart: heavy.fromStart,
        stopS: heavy.stopS,
        fast,
        tram,
        stretchSpeeds: heavy.stretchSpeeds,
        shift,
        passed: heavy.passed,
        reversals: heavy.reversals,
        endTurns: heavy.endTurns,
        ways: heavy.ways,
        oneStops: heavy.oneStops,
        ...(heavy.stopOnly.length ? { stopOnly: heavy.stopOnly } : {}),
        hand: heavy.hand,
        loopKind,
        oneWay,
        oneWayByTrack,
        limit,
        ...(fareDist ? { fareDist: true as const } : {}),
        ...(weekend.length ? { weekend } : {}),
        restTph,
        tphP, periods, periodsOwn, hours, hoursOwn,
        ...(rows ? { rows } : {}),
        ...(pairs === 1 || pairs === 2 ? { pairs } : {}),
        ...(twinOf >= 0 ? { twinOf } : {}),
        ...(r.busRow ? { bus: r.busRow } : {}),
      };
      kept.set(id, { row, raw, heavy, extra, own });
      out.push(row);
    }
    if (kept.size > seen.size) for (const id of kept.keys()) if (!seen.has(id)) kept.delete(id);
    for (const id of refresh) this.phasePending.add(id);
    this.phaseSeen = seen;
    return out;
  }

  /** the touched nodes and edges (`take_touched`), for the regions to move */
  private touchedIds(t: ArrayLike<number>): { nodes: Set<number>; edges: Set<number> } {
    const edges = new Set<number>(), nodes = new Set<number>(), ne = t[0];
    for (let k = 1; k <= ne; k++) edges.add(t[k]);
    for (let k = 1 + ne; k < t.length; k++) nodes.add(t[k]);
    return { nodes, edges };
  }

  /** T-4130: whether a kept row still holds what `edges_info` (`nodes_info`) gives it at offset `i`. Some of those numbers
   * derive from every line (a junction's crossing moves), so a record outside the read's scope can still move. */
  private static edgeBulkSame(r: ClockEdge, ei: ArrayLike<number>, i: number): boolean {
    return r.a === ei[i + 1] && r.b === ei[i + 2] && r.tracks === ei[i + 3] && r.built === (ei[i + 4] === 1) && Object.is(r.lengthM, ei[i + 5]) && Object.is(r.cost, ei[i + 6])
      && Object.is(r.vminKmh, ei[i + 7]) && r.levelMin === ei[i + 8] && r.levelMax === ei[i + 9];
  }
  private static nodeBulkSame(r: ClockNode, ni: ArrayLike<number>, i: number): boolean {
    const f = ni[i + 6];
    return r.x === ni[i + 1] && r.y === ni[i + 2] && r.level === ni[i + 3] && r.ports === ni[i + 4] && r.builtPorts === ni[i + 5] && r.flying === ((f & 1) === 1)
      && !!r.crosses === ((f & 2) === 2) && (r.unusable ?? 0) === ((f >> 2) & 3) && r.platform === ni[i + 7] && r.stationBuilt === (ni[i + 8] === 1) && Object.is(r.heading, ni[i + 9]);
  }

  /** T-4113: a set or map from a bulk call against the last read's: the ids whose membership or value moved */
  private static moved(was: Map<number, unknown> | Set<number> | undefined, now: Map<number, unknown> | Set<number>, into: Set<number>) {
    if (!was) return;
    const get = (m: Map<number, unknown> | Set<number>, k: number) => (m instanceof Map ? m.get(k) : m.has(k));
    for (const k of now.keys()) if (get(was, k) !== get(now, k)) into.add(k);
    for (const k of was.keys()) if (!now.has(k)) into.add(k);
  }
  private lastSets = new Map<string, Map<number, unknown> | Set<number>>();
  /** the last read's model nodes, folded, and whether this read changed any */
  private lastNodes: ClockNode[] | null = null;
  private lastFoldKey = "";

  /** whether any of these points (x, y each) is in a region the read takes again */
  private reaches(t: Taken, pts: Float64Array): boolean {
    if (t.all) return true;
    for (let k = 0; k + 1 < pts.length; k += 2) if (this.regions.isTaken(t, pts[k], pts[k + 1])) return true;
    return false;
  }

  /** Every line's phase tables laid end to end, as `TrackApi.phases` answers, read after `lines`
   * (the trials that price a planned line or the construct-all quote, asked between states, note the
   * lines they recompute as changed: they are read here, and again by the next `lines`). The same
   * array as last time while no line's tables changed; never to be transferred or changed.
   * `meta` (`profile_meta`, read with them): T-3896, tables laid out to another length than it counts
   * are read whole, since every line's after the first wrong one would be read from the wrong place. */
  phases(a: TrackApi, meta?: Float32Array): Float32Array {
    const ids = new Set(this.phasePending);
    this.phasePending.clear();
    const moved = a.peek_changed_lines();
    if (moved.length && moved[0] === ALL) for (const id of this.phaseSeen) ids.add(id);
    else for (const id of moved) if (this.phaseSeen.has(id)) ids.add(id);
    this.phaseReads = ids.size;
    if (ids.size > StateRows.wholePhases) this.splitPhases(a);
    else {
      for (const id of ids) this.readPhases(a, id);
      for (const id of this.phaseTables.keys()) if (!this.phaseSeen.has(id)) this.dropPhases(id);
    }
    const out = this.phaseStale || !this.phaseAll ? this.layOutPhases() : this.phaseAll;
    if (!meta) return out;
    let n = 0;
    for (let k = 1; k < meta.length; k += 4) n += meta[k];
    if (n * 4 === out.length) return out;
    console.error(`T-3896: the phase tables laid out by line are ${out.length / 4} phases against the ${n} counted: read whole`);
    this.splitPhases(a);
    return this.phaseAll!;
  }

  /** A line's phase tables, kept if they are the same. */
  private readPhases(a: TrackApi, id: number) {
    const t = a.line_phases(id), old = this.phaseTables.get(id);
    if (old ? sameNumbers(old, t) : t.length === 0) return;
    this.phaseStale = true;
    this.phaseDirty.add(id);
    if (old?.length !== t.length) this.phaseLayout = true;
    if (t.length) this.phaseTables.set(id, t); else this.phaseTables.delete(id);
  }

  private dropPhases(id: number) {
    this.phaseTables.delete(id);
    this.phaseStale = this.phaseLayout = true;
  }

  /** All the lines' tables in one call, for an edit that changed many of them (a call each costs
   * more past `wholePhases`): the answer is that array, and each line's table a view of it. */
  private splitPhases(a: TrackApi) {
    const whole = a.phases(), meta = a.profile_meta();
    this.phaseTables.clear();
    for (const id of this.phaseSeen) {
      const m = id * META_PER_LINE;
      if (m + META_PER_LINE > meta.length) continue;
      let count = 0;
      for (let p = 0; p < PROFILES_PER_LINE; p++) count += meta[m + p * 4 + 1];
      if (count > 0) this.phaseTables.set(id, whole.subarray(meta[m] * 4, (meta[m] + count) * 4));
    }
    this.phaseAll = whole;
    this.phaseStale = this.phaseLayout = false;
    this.phaseDirty.clear();
  }

  /** The tables end to end in line order: a copy of the last array with the changed tables written
   * over theirs when none changed in length, otherwise all of them laid out again. */
  private layOutPhases(): Float32Array {
    const ids = [...this.phaseTables.keys()].sort((x, y) => x - y);
    let out: Float32Array;
    if (this.phaseAll && !this.phaseLayout) {
      out = this.phaseAll.slice();
      let at = 0;
      for (const id of ids) {
        const t = this.phaseTables.get(id)!;
        if (this.phaseDirty.has(id)) out.set(t, at);
        at += t.length;
      }
    } else {
      let n = 0;
      for (const id of ids) n += this.phaseTables.get(id)!.length;
      out = new Float32Array(n);
      let at = 0;
      for (const id of ids) { const t = this.phaseTables.get(id)!; out.set(t, at); at += t.length; }
    }
    this.phaseAll = out;
    this.phaseStale = this.phaseLayout = false;
    this.phaseDirty.clear();
    return out;
  }

  private heavy(a: TrackApi, id: number, ok: boolean): Heavy {
    const part = a.line_part(id);
    return {
      ...(part.length === 4
        ? { stops: [...a.line_run_stops(id)], path: [...a.line_run_path(id)], planStops: [...a.line_stops(id)], planPath: [...a.line_path(id)], stopStart: part[0], pathStart: part[2] }
        : { stops: [...a.line_stops(id)], path: [...a.line_path(id)], planStops: [], planPath: [], stopStart: 0, pathStart: 0 }),
      name: a.line_name(id),
      letter: a.line_symbol(id),
      group: a.line_group(id),
      groupName: a.line_group_name(id),
      groupSymbol: a.line_group_symbol(id),
      groupColour: ((c) => (c < 0 ? "" : hex(c)))(a.line_group_colour(id)),
      outer: [...a.line_group_outer(id)].map((group, k) => ({ group, name: a.line_group_outer_name(id, k), symbol: a.line_group_outer_symbol(id, k), colour: ((c) => (c < 0 ? "" : hex(c)))(a.line_group_outer_colour(id, k)) })),
      back: [...a.line_back_path(id)],
      backStops: [...a.line_back_stops(id)],
      fromStart: [0, 1, 2].map((lev) => [...a.stop_times(id, 0, lev)]),
      stopS: [[...a.line_stop_s(id, 0)], [...a.line_stop_s(id, 1)]],
      stretchSpeeds: ok ? [...a.line_stretch_speeds(id)] : [],
      passed: [...a.passed_stations(id)],
      reversals: [...a.line_reversals(id)],
      endTurns: [...a.line_end_turns(id)],
      ways: [...a.line_stop_ways(id)],
      oneStops: [...a.line_one_stops(id)],
      stopOnly: [...a.line_stop_only(id)],
      hand: a.line_hand(id),
    };
  }
}
