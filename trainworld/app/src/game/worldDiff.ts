// T-4150: what changed between two worlds, record by record, and values kept per record across worlds.
//
// The clock client keeps each record's view object while its row is the same (workers/clockClient.ts `toWorld`, over the
// worker's row deltas, T-1029), so a record whose object is the same is the same record: an edit's changes are read off
// identity, a pointer compare a record, and a reader redoes only the records that changed instead of the whole map.
// `worldDiff` takes any two worlds (a lazy computed that skipped a state, or `moneyWorld` holding an older one, diffs from
// the world it last read); a new game's records are new objects, so its ids count as changed even where they repeat.

import { PATCH_MODE } from "./patchMode";
import type { LineId, LineInput, StationInput, TrackEdge, TrackNode, WorldView } from "./types";

/** Each changed record's id with its object before and after (undefined where added or removed). */
export type Changes<K, T> = ReadonlyMap<K, readonly [T | undefined, T | undefined]>;

export interface WorldDiff {
  /** stations added, removed or changed */
  stations: Changes<string, StationInput>;
  /** lines whose row (`save.lines`) or figures (`lineStats`) changed, added or removed (by id; the objects are no help
   * here, a line being two) */
  lines: ReadonlySet<LineId>;
  edges: Changes<number, TrackEdge>;
  nodes: Changes<number, TrackNode>;
  /** the passages (`save.links`, `apart`) are other lists */
  links: boolean;
  /** the station buildings are another list */
  buildings: boolean;
  /** T-4150 stage 2: the lines a per-line reader reads again: those in `lines` and those stopping at a station that changed */
  reading: ReadonlySet<LineId>;
}

/** The records of `a` and `b` whose objects differ, or that only one has. Both lists in ascending id order (the model's)
 * are walked side by side; one that is not is gone through by id from where it breaks the order, with the records the
 * walk found in one list only (one of them can be in the other's rest). */
function diffList<K extends string | number, T extends { id: K }>(a: Iterable<T>, b: Iterable<T>, out: Map<K, readonly [T | undefined, T | undefined]>): void {
  const ia = a[Symbol.iterator](), ib = b[Symbol.iterator]();
  let x = ia.next(), y = ib.next();
  let lastA: K | undefined, lastB: K | undefined;
  const key = (k: K) => (typeof k === "number" ? k : Number(k));
  const outOfOrder = (k: K | undefined, last: K | undefined) => k !== undefined && (Number.isNaN(key(k)) || (last !== undefined && key(k) <= key(last)));
  while (!x.done || !y.done) {
    const ka = x.done ? undefined : x.value.id, kb = y.done ? undefined : y.value.id;
    if (outOfOrder(ka, lastA) || outOfOrder(kb, lastB)) {
      byId(x, ia, y, ib, out);
      return;
    }
    if (ka !== undefined && kb !== undefined && ka === kb) {
      if (x.value !== y.value) out.set(kb, [x.value as T, y.value as T]);
      lastA = ka; lastB = kb;
      x = ia.next(); y = ib.next();
    } else if (kb === undefined || (ka !== undefined && key(ka) < key(kb))) {
      out.set(ka!, [x.value as T, undefined]);
      lastA = ka;
      x = ia.next();
    } else {
      out.set(kb, [undefined, y.value as T]);
      lastB = kb;
      y = ib.next();
    }
  }
}
/** the rest of two lists by id, and the records `out` has from one list only */
function byId<K, T extends { id: K }>(x: IteratorResult<T>, ia: Iterator<T>, y: IteratorResult<T>, ib: Iterator<T>, out: Map<K, readonly [T | undefined, T | undefined]>) {
  const was = new Map<K, T>(), now: T[] = [];
  for (const [k, [p, q]] of out) {
    if (p && q) continue;
    if (p) was.set(k, p);
    else now.push(q!);
    out.delete(k);
  }
  for (let r = x; !r.done; r = ia.next()) was.set(r.value.id, r.value);
  for (let r = y; !r.done; r = ib.next()) now.push(r.value);
  for (const r of now) {
    const v = was.get(r.id);
    if (v !== r) out.set(r.id, [v, r]);
    was.delete(r.id);
  }
  for (const [k, v] of was) out.set(k, [v, undefined]);
}

/** T-4150 stage 2: the edges and nodes the page was told may differ from one world it built to the next (the state's
 * `touched`, workers/clockClient.ts `toWorld`); null for a kind not known (a whole state), which is walked */
export interface StepIds { edges: ReadonlySet<number> | null; nodes: ReadonlySet<number> | null }
/** the worlds one diff composes at most before it walks */
const MAX_STEPS = 32;
/** Each world built in turn gets the next number, and the ring holds what each of the last `MAX_STEPS` steps named (by
 * number, so no world keeps an older one alive); a world built from one that is not the last starts a new run. */
const serial = new WeakMap<WorldView, number>();
let lastSerial = 0;
const ring: (StepIds | null)[] = new Array(MAX_STEPS).fill(null);
/** `to` was built from `from` with only the records `ids` names possibly other objects (null: any) */
export function noteStep(from: WorldView | null, to: WorldView, ids: StepIds | null) {
  if (serial.has(to)) return;
  const chained = !!from && serial.get(from) === lastSerial;
  serial.set(to, ++lastSerial);
  ring[lastSerial % MAX_STEPS] = chained ? ids : null;
}
/** T-4284: worlds built from another with a few rows patched (the tools' click world, workers/clockClient.ts `layPatch`),
 * off the numbered chain: the world each was built from and the ids its rows may differ by */
const sides = new WeakMap<WorldView, { from: WorldView; ids: StepIds }>();
export function noteSide(from: WorldView, to: WorldView, ids: StepIds) {
  sides.set(to, { from, ids });
}
/** the world a click world was built from (`noteSide`), else null */
export const sideOf = (w: WorldView): WorldView | null => sides.get(w)?.from ?? null;
/** The ids of kind `k` that may differ from `a` to `b` when `b` was built from `a` through a few steps, else null. */
function stepIds(a: WorldView, b: WorldView, k: keyof StepIds): Set<number> | null {
  const side = sides.get(b);
  if (side) {
    const ids = side.from === a ? side.ids[k] : null;
    return ids ? new Set(ids) : null;
  }
  const sa = serial.get(a), sb = serial.get(b);
  if (sa === undefined || sb === undefined || sb <= sa || sb - sa > MAX_STEPS - 1 || lastSerial - sa > MAX_STEPS - 1) return null;
  const out = new Set<number>();
  for (let n = sa + 1; n <= sb; n++) {
    const ids = ring[n % MAX_STEPS]?.[k];
    if (!ids) return null;
    for (const id of ids) out.add(id);
  }
  return out;
}
/** whether a list's ids ascend (the model's order), found once per list */
const ascending = new WeakMap<readonly unknown[], boolean>();
const indexes = new WeakMap<readonly unknown[], Map<number, unknown>>();
/** Whether `list`'s `key`s ascend, found once per list. */
export function ascendingBy<T>(list: readonly T[], key: (x: T) => number): boolean {
  let up = ascending.get(list);
  if (up === undefined) {
    up = true;
    for (let i = 1; i < list.length; i++) if (!(key(list[i - 1]) < key(list[i]))) { up = false; break; }
    ascending.set(list, up);
  }
  return up;
}
/** The record with id `id` (`key` of it) in `list`: a binary search in a list in id order, else by a map made once. */
function findIn<T>(list: readonly T[], id: number, key: (x: T) => number): T | undefined {
  if (!ascendingBy(list, key)) {
    let m = indexes.get(list);
    if (!m) indexes.set(list, (m = new Map(list.map((x) => [key(x), x]))));
    return m.get(id) as T | undefined;
  }
  let lo = 0, hi = list.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (key(list[mid]) < id) lo = mid + 1; else hi = mid; }
  return lo < list.length && key(list[lo]) === id ? list[lo] : undefined;
}
/** T-4284: the edge with id `id` of a world, without a pass over its edges (found as `worldDiff` finds them) */
export const edgeOf = (w: WorldView, id: number): TrackEdge | undefined => findIn(w.edges, id, (e) => e.id);
/** DEV: how the stations, edges and nodes that changed were found: by the steps' ids (and how many ids looked up), or by
 * walking both lists (and how many records); the lines are always walked (a few hundred pointers) */
export const diffStats = { ids: 0, looked: 0, walks: 0, walked: 0 };
if (import.meta.env?.DEV && typeof window !== "undefined") (window as unknown as { twDiffStats: typeof diffStats }).twDiffStats = diffStats;
const walkCount = (n: number) => { diffStats.walks++; diffStats.walked += n; };
/** the changes among `ids` from `a` to `b` */
function diffIds<T>(ids: Iterable<number>, get: (side: 0 | 1, id: number) => T | undefined, out: Map<number, readonly [T | undefined, T | undefined]>) {
  diffStats.ids++;
  for (const id of ids) {
    diffStats.looked++;
    const x = get(0, id), y = get(1, id);
    if (x !== y) out.set(id, [x, y]);
  }
}

/** T-4150 stage 2: each station's lines (ids), for one world at a time: moved to another by the lines that differ */
let stopIndex: { w: WorldView; at: Map<string, Set<LineId>> } | null = null;
function stopsAtOf(w: WorldView, d: (a: WorldView) => WorldDiff | null): Map<string, Set<LineId>> {
  const k = stopIndex;
  if (k && k.w === w) return k.at;
  const was = k ? lineRowsOf(d(k.w)) : null;
  if (!k || !was || freshNow()) {
    const at = new Map<string, Set<LineId>>();
    for (const l of w.save.lines) for (const s of l.stops) (at.get(s) ?? at.set(s, new Set()).get(s)!).add(l.id);
    if (!freshNow()) stopIndex = { w, at };
    return at;
  }
  for (const [id, [x]] of was) for (const s of x?.stops ?? []) k.at.get(s)?.delete(id);
  for (const [id, [, y]] of was) for (const s of y?.stops ?? []) (k.at.get(s) ?? k.at.set(s, new Set()).get(s)!).add(id);
  k.w = w;
  return k.at;
}
/** each diff's line rows that differ (`save.lines`), with their objects before and after */
const lineRowsGet = new WeakMap<WorldDiff, () => Changes<LineId, LineInput>>();
const lineRowsOf = (d: WorldDiff | null) => (d ? lineRowsGet.get(d)?.() ?? null : null);

/** by the older world, then the newer: a diff holds both, and keyed so no world keeps an older one alive */
const memo = new WeakMap<WorldView, WeakMap<WorldView, WorldDiff>>();

/** What changed from world `a` to world `b`; null with no `a` (read everything). Each kind is worked out when first read,
 * and kept while both worlds are. The edges, nodes and stations of a world built from `a` through a few
 * steps are looked up by the ids the steps name; of any other two worlds, both lists are walked. */
export function worldDiff(a: WorldView | null | undefined, b: WorldView): WorldDiff | null {
  if (!a) return null;
  const kept = memo.get(a)?.get(b);
  if (kept) return kept;
  const once = <T>(f: () => T) => { let v: T | undefined; return () => (v ??= f()); };
  const ends = [a, b] as const;
  const stations = once(() => {
    const m = new Map<string, readonly [StationInput | undefined, StationInput | undefined]>();
    if (a.save.stations === b.save.stations) return m;
    const ids = stepIds(a, b, "nodes");
    if (!ids) { walkCount(a.save.stations.length + b.save.stations.length); diffList(a.save.stations, b.save.stations, m); return m; }
    const got = new Map<number, readonly [StationInput | undefined, StationInput | undefined]>();
    diffIds(ids, (k, id) => findIn(ends[k].save.stations, id, (s) => s.num), got);
    for (const [id, p] of got) m.set(String(id), p);
    if (checks.on) checkStep("stations", m, () => { const w = new Map(); diffList(a.save.stations, b.save.stations, w); return w; });
    return m;
  });
  const lineRows = once(() => {
    const ls = new Map<string, readonly [LineInput | undefined, LineInput | undefined]>();
    if (a.save.lines !== b.save.lines) diffList(a.save.lines, b.save.lines, ls);
    return ls;
  });
  const lines = once(() => {
    const out = new Set<string>(lineRows().keys());
    if (a.lineStats !== b.lineStats) {
      for (const id in b.lineStats) if (a.lineStats[id] !== b.lineStats[id]) out.add(id);
      for (const id in a.lineStats) if (!(id in b.lineStats)) out.add(id);
    }
    return out;
  });
  const edges = once(() => {
    const m = new Map<number, readonly [TrackEdge | undefined, TrackEdge | undefined]>();
    if (a.edges === b.edges) return m;
    const ids = stepIds(a, b, "edges");
    if (!ids) { walkCount(a.edges.length + b.edges.length); diffList(a.edges, b.edges, m); return m; }
    diffIds(ids, (k, id) => findIn(ends[k].edges, id, (e) => e.id), m);
    if (checks.on) checkStep("edges", m, () => { const w = new Map(); diffList(a.edges, b.edges, w); return w; });
    return m;
  });
  const nodes = once(() => {
    const m = new Map<number, readonly [TrackNode | undefined, TrackNode | undefined]>();
    if (a.nodes === b.nodes) return m;
    const ids = stepIds(a, b, "nodes");
    if (!ids) { walkCount(a.nodes.size + b.nodes.size); diffList(a.nodes.values(), b.nodes.values(), m); return m; }
    diffIds(ids, (k, id) => ends[k].nodes.get(id), m);
    if (checks.on) checkStep("nodes", m, () => { const w = new Map(); diffList(a.nodes.values(), b.nodes.values(), w); return w; });
    return m;
  });
  const reading = once(() => {
    const out = new Set(lines());
    if (stations().size) {
      const at = stopsAtOf(b, (x) => worldDiff(x, b));
      for (const id of stations().keys()) for (const l of at.get(id) ?? []) out.add(l);
    }
    return out;
  });
  const d: WorldDiff = {
    get stations() { return stations(); },
    get lines() { return lines(); },
    get edges() { return edges(); },
    get nodes() { return nodes(); },
    get reading() { return reading(); },
    links: a.save.links !== b.save.links || a.save.apart !== b.save.apart,
    buildings: a.save.buildings !== b.save.buildings,
  };
  lineRowsGet.set(d, lineRows);
  (memo.get(a) ?? memo.set(a, new WeakMap()).get(a)!).set(b, d);
  return d;
}
/** DEV: a diff looked up by ids against the walk of both lists */
function checkStep(kind: string, got: Map<unknown, unknown>, walk: () => Map<unknown, unknown>) {
  const was = { ...diffStats }, whole = walk();
  Object.assign(diffStats, was);
  if (!sameBuilt(got, whole)) console.error(`T-4150 check: the ${kind} changed, looked up by the state's touched ids, differ from a walk of both worlds`);
}

const checks = { on: PATCH_MODE === "check" };
/** `?patch=check` (as T-1029, T-1380), or `twPatchChecks(true)` in a DEV page: readers that patch from a diff also build
 * in full and compare */
export const checkPatches = () => checks.on;
if (import.meta.env?.DEV && typeof window !== "undefined") (window as unknown as { twPatchChecks: (on: boolean) => void }).twPatchChecks = (on) => { checks.on = on; };

let fresh = 0;
/** inside `freshly`: memos of whole results are not read either */
export const freshNow = () => fresh > 0;
/** `f` with every `PerRecord` making its values again (a check's whole build) */
export function freshly<T>(f: () => T): T {
  fresh++;
  try { return f(); } finally { fresh--; }
}

const sameDeps = (a: readonly unknown[], b: readonly unknown[]) => a.length === b.length && a.every((x, i) => x === b[i] || (x !== x && b[i] !== b[i]));

/** Values kept per record id with the inputs they were made from (objects, compared by identity, or numbers): `get` makes
 * one again only when an input differs. A pass (`begin`, the `get`s, `end`) drops the ids it did not ask for. */
export class PerRecord<K, V> {
  private m = new Map<K, { deps: readonly unknown[]; v: V; pass: number }>();
  private pass = 0;
  begin() { this.pass++; }
  get(id: K, deps: readonly unknown[], make: () => V): V {
    const e = this.m.get(id);
    if (e && !fresh && sameDeps(e.deps, deps)) {
      e.pass = this.pass;
      return e.v;
    }
    const v = make();
    if (!fresh) this.m.set(id, { deps, v, pass: this.pass });
    return v;
  }
  /** T-4150 stage 2: the value kept for `id`, its inputs not compared (the caller knows they are the same), counted as asked for */
  keep(id: K): V | undefined {
    const e = fresh ? undefined : this.m.get(id);
    if (e) e.pass = this.pass;
    return e?.v;
  }
  end() {
    if (fresh) return;
    for (const [k, e] of this.m) if (e.pass !== this.pass) this.m.delete(k);
  }
  /** A new pass where callers ask for some records only: past `2 live + 64` values, those the last pass left are dropped. */
  trim(live: number) {
    if (!fresh && this.m.size > 2 * live + 64) for (const [k, e] of this.m) if (e.pass !== this.pass) this.m.delete(k);
    this.pass++;
  }
}

/** T-4150 stage 2: a value per line kept from the world a pass last read: `begin(w, same)` makes again only the lines the
 * diff from that world reads again (`reading`), every line when `same` is false (another input changed); a pass keeps only
 * the lines it asked for. Inside `freshly` every value is made and nothing kept. */
export class PerLine<V> {
  private w: WorldView | null = null;
  /** the world the pass under way reads, kept as `w` once it ends (a pass that never ends leaves the last one's) */
  private at: WorldView | null = null;
  private m = new Map<LineId, V>();
  private next = new Map<LineId, V>();
  private reading: ReadonlySet<LineId> | null = null;
  begin(w: WorldView, same = true) {
    if (fresh) return;
    this.reading = same && this.w ? worldDiff(this.w, w)?.reading ?? null : null;
    this.at = w;
    this.next = new Map();
  }
  get(id: LineId, make: () => V): V {
    if (fresh) return make();
    let v = this.reading && !this.reading.has(id) ? this.m.get(id) : undefined;
    if (v === undefined) v = make();
    this.next.set(id, v);
    return v;
  }
  end() {
    if (fresh) return;
    this.m = this.next;
    this.w = this.at;
  }
}

/** DEV: two built values the same, Maps and Sets by their entries (numbers exactly) */
export function sameBuilt(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a === "number" && typeof b === "number") return a !== a && b !== b;
  if (!a || !b || typeof a !== "object" || typeof b !== "object") return false;
  if (a instanceof Map || b instanceof Map) {
    if (!(a instanceof Map) || !(b instanceof Map) || a.size !== b.size) return false;
    for (const [k, v] of a) if (!b.has(k) || !sameBuilt(v, b.get(k))) return false;
    return true;
  }
  if (a instanceof Set || b instanceof Set) return a instanceof Set && b instanceof Set && a.size === b.size && [...a].every((x) => b.has(x));
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && sameBuilt((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}
