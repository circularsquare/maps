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
//    recomputed (`take_changed_lines`), which is every line an edit can change them for.
// The result is the same as `new StateRows().read(...)` on the same model, rows field for field
// (checks/t1382-rows.mjs; `?patch=check` compares them in the page).
import type { TrackApi } from "../wasm/track/sim";
import type { ClockEdge, ClockLine, ClockNode } from "./protocol";

/** what the model answers for "every one" in `take_changed_lines` and `take_name_changes` */
const ALL = 0xffffffff;

export interface RowEnv {
  /** world metres to longitude and latitude */
  lngLat(x: number, y: number): [number, number];
  /** hours of each demand level in a day, for `line_upkeep` */
  levelHours: Float64Array;
  /** a bus line's row, and the nodes for its stops that are no station (`nodes` is the list so far) */
  busRow(a: TrackApi, id: number, nodes: ClockNode[]): NonNullable<ClockLine["bus"]>;
}

/** What a line row holds that costs several model calls and changes only when the line's service or
 * net does. `passed` is in the state, not asked for after it, so the line panel never draws new stops
 * with old gaps (T-648). */
interface Heavy {
  name: string;
  letter: string;
  stops: number[];
  path: number[];
  fromStart: number[][];
  stopS: [number[], number[]];
  stretchSpeeds: number[];
  passed: number[];
  reversals: number[];
  endTurns: number[];
}

/** a kept line: its row, and the numbers the row was made from */
interface LineKept {
  row: ClockLine;
  raw: Float64Array;
  heavy: Heavy;
  extra: number[];
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
  loopKind: number;
  shift: number;
  upkeep: number;
  busRow: ClockLine["bus"];
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
  private names = new Map<number, string>();
  private lineKept = new Map<number, LineKept>();

  /** Forget every row: the next read is a full one. */
  reset() {
    this.owner = null;
    this.edgeRows.clear();
    this.nodeRows.clear();
    this.names.clear();
    this.lineKept.clear();
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
    if (changed.length && changed[0] === ALL) this.names.clear();
    else for (const id of changed) this.names.delete(id);
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
  read(a: TrackApi, env: RowEnv, frameKey: unknown, ups: Map<number, number> = upgradesOf(a)): { edges: ClockEdge[]; nodes: ClockNode[]; lines: ClockLine[] } {
    this.bind(a, frameKey);
    const edges = this.edges(a), nodes = this.nodes(a, env, ups);
    return { edges, nodes, lines: this.lines(a, env, nodes) };
  }

  edges(a: TrackApi): ClockEdge[] {
    const ei = a.edges_info();
    const offEdges = new Set<number>(a.off_edges());
    const hsrEdges = new Set<number>(a.hsr_edges());
    const crossoverIds = a.crossover_edges(), crossoverPartners = a.crossover_partners();
    const crossovers = new Map<number, number>(Array.from(crossoverIds, (id, k) => [id, crossoverPartners[k] === 0xffffffff ? -1 : crossoverPartners[k]]));
    const out: ClockEdge[] = [];
    const kept = this.edgeRows;
    const seen = new Set<number>();
    for (let i = 0; i < ei.length; i += 10) {
      const id = ei[i];
      const off = offEdges.has(id), hsr = hsrEdges.has(id), crossover = crossovers.has(id), partner = crossovers.get(id) ?? -1;
      const tracks = ei[i + 3] as 1 | 2, built = ei[i + 4] === 1;
      let r = kept.get(id);
      if (!r || r.a !== ei[i + 1] || r.b !== ei[i + 2] || r.tracks !== tracks || r.built !== built || r.off !== off || r.hsr !== hsr || r.crossover !== crossover || r.partner !== partner
        || !Object.is(r.lengthM, ei[i + 5]) || !Object.is(r.cost, ei[i + 6]) || !Object.is(r.vminKmh, ei[i + 7]) || r.levelMin !== ei[i + 8] || r.levelMax !== ei[i + 9]) {
        r = { id, a: ei[i + 1], b: ei[i + 2], tracks, built, off, hsr, crossover, partner, lengthM: ei[i + 5], cost: ei[i + 6], vminKmh: ei[i + 7], levelMin: ei[i + 8], levelMax: ei[i + 9] };
        kept.set(id, r);
      }
      seen.add(id);
      out.push(r);
    }
    if (kept.size !== seen.size) for (const id of kept.keys()) if (!seen.has(id)) kept.delete(id);
    return out;
  }

  nodes(a: TrackApi, env: RowEnv, ups: Map<number, number>): ClockNode[] {
    this.syncNames(a);
    const ni = a.nodes_info();
    const offStations = new Set<number>(a.off_stations());
    const out: ClockNode[] = [];
    const kept = this.nodeRows;
    const seen = new Set<number>();
    for (let i = 0; i < ni.length; i += 10) {
      const id = ni[i];
      const platform = ni[i + 7];
      const stationBuilt = ni[i + 8] === 1, stationOff = offStations.has(id), upgrades = ups.get(id) ?? 0, flying = ni[i + 6] === 1;
      const name = platform > 0 ? this.name(a, id) : "";
      let r = kept.get(id);
      if (!r || r.x !== ni[i + 1] || r.y !== ni[i + 2] || r.level !== ni[i + 3] || r.ports !== ni[i + 4] || r.builtPorts !== ni[i + 5] || r.flying !== flying || r.platform !== platform
        || r.stationBuilt !== stationBuilt || r.stationOff !== stationOff || r.upgrades !== upgrades || r.name !== name || !Object.is(r.heading, ni[i + 9])) {
        const [lng, lat] = env.lngLat(ni[i + 1], ni[i + 2]);
        r = { id, x: ni[i + 1], y: ni[i + 2], lng, lat, level: ni[i + 3], ports: ni[i + 4], builtPorts: ni[i + 5], flying, platform, stationBuilt, stationOff, upgrades, name, heading: ni[i + 9] };
        kept.set(id, r);
      }
      seen.add(id);
      out.push(r);
    }
    if (kept.size !== seen.size) for (const id of kept.keys()) if (!seen.has(id)) kept.delete(id);
    // a station no longer there keeps no name
    if (this.names.size > seen.size) for (const id of this.names.keys()) if (!seen.has(id)) this.names.delete(id);
    return out;
  }

  /** `nodes` must have been read just before, for the bus stops that join it. Everything but the build
   * and train costs is read first and the costs after: pricing a planned line runs a trial that
   * settles headway lines a little differently each time (`line_build_cost`), so rows read between
   * two trials would depend on which lines were read before. The costs are asked in line order. */
  lines(a: TrackApi, env: RowEnv, nodes?: ClockNode[]): ClockLine[] {
    const changed = a.take_changed_lines();
    const all = changed.length > 0 && changed[0] === ALL;
    const dirty = new Set<number>(all ? [] : changed);
    const li = a.lines_info();
    const lu = a.line_upkeep(env.levelHours), upkeepOf = new Map<number, number>();
    for (let k = 0; k + 1 < lu.length; k += 2) upkeepOf.set(lu[k], lu[k + 1]);
    const kept = this.lineKept;
    const seen = new Set<number>();
    const read: LineRead[] = [];
    for (let i = 0; i < li.length; i += 22) {
      const id = li[i];
      seen.add(id);
      const ok = li[i + 2] === 1, running = li[i + 3] === 1;
      const bus = a.is_bus(id);
      const prev = kept.get(id);
      const fresh = all || bus || !prev || dirty.has(id) || prev.row.ok !== ok || prev.row.running !== running;
      read.push({
        i, id, ok, running, bus, prev, fresh,
        heavy: fresh ? this.heavy(a, id, ok) : prev!.heavy,
        auto: a.line_auto(id), fast: a.line_fast(id), loopKind: a.line_loop(id), shift: a.line_shift(id), upkeep: upkeepOf.get(id) ?? 0,
        busRow: bus ? env.busRow(a, id, nodes ?? []) : undefined,
      });
    }
    const out: ClockLine[] = [];
    for (const r of read) {
      const { i, id, ok, running, prev, fresh, heavy, auto, fast, loopKind, shift, upkeep } = r;
      const buildCost = ok && !running ? a.line_build_cost(id) : 0;
      const trainCost = ok && !running ? a.line_train_cost(id) : 0;
      // the numbers behind the cheap fields: when all are as kept (and the heavy parts are), the row is the kept one
      const raw = li.slice(i, i + 22);
      const extra = [upkeep, buildCost, trainCost, fast ? 1 : 0, loopKind, shift, ...auto];
      if (!fresh && prev && sameNumbers(raw, prev.raw) && sameNumbers(extra, prev.extra)) {
        out.push(prev.row);
        continue;
      }
      const t3 = (k: number) => [li[i + k], li[i + k + 1], li[i + k + 2]] as [number, number, number];
      const row: ClockLine = {
        upkeep,
        id, name: heavy.name, letter: heavy.letter, colour: hex(li[i + 1]), ok, running, broken: li[i + 4] === 1, tph: t3(5), cars: li[i + 21] || li[i + 8], carsSet: li[i + 8],
        lengthM: li[i + 9], roundTrip: t3(10), trains: t3(13), auto: autoOf(auto), delay: t3(16), dwell: li[i + 19], turnaround: li[i + 20],
        stops: heavy.stops, path: heavy.path,
        fromStart: heavy.fromStart,
        stopS: heavy.stopS,
        fast,
        stretchSpeeds: heavy.stretchSpeeds,
        buildCost,
        trainCost,
        shift,
        passed: heavy.passed,
        reversals: heavy.reversals,
        endTurns: heavy.endTurns,
        loopKind,
        ...(r.busRow ? { bus: r.busRow } : {}),
      };
      if (r.bus) kept.delete(id);
      else kept.set(id, { row, raw, heavy, extra });
      out.push(row);
    }
    if (kept.size > seen.size) for (const id of kept.keys()) if (!seen.has(id)) kept.delete(id);
    return out;
  }

  private heavy(a: TrackApi, id: number, ok: boolean): Heavy {
    return {
      name: a.line_name(id),
      letter: a.line_symbol(id),
      stops: [...a.line_stops(id)],
      path: [...a.line_path(id)],
      fromStart: [0, 1, 2].map((lev) => [...a.stop_times(id, 0, lev)]),
      stopS: [[...a.line_stop_s(id, 0)], [...a.line_stop_s(id, 1)]],
      stretchSpeeds: ok ? [...a.line_stretch_speeds(id)] : [],
      passed: [...a.passed_stations(id)],
      reversals: [...a.line_reversals(id)],
      endTurns: [...a.line_end_turns(id)],
    };
  }
}
