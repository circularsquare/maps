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

const memo = new WeakMap<WorldView, { from: WorldView; d: WorldDiff }>();

/** What changed from world `a` to world `b`; null with no `a` (read everything). Each kind is worked out when first read,
 * and kept for the last `a` asked of each `b`. */
export function worldDiff(a: WorldView | null | undefined, b: WorldView): WorldDiff | null {
  if (!a) return null;
  const kept = memo.get(b);
  if (kept && kept.from === a) return kept.d;
  const once = <T>(f: () => T) => { let v: T | undefined; return () => (v ??= f()); };
  const stations = once(() => { const m = new Map<string, readonly [StationInput | undefined, StationInput | undefined]>(); if (a.save.stations !== b.save.stations) diffList(a.save.stations, b.save.stations, m); return m; });
  const lines = once(() => {
    const out = new Set<string>();
    if (a.save.lines !== b.save.lines) {
      const ls = new Map<string, readonly [LineInput | undefined, LineInput | undefined]>();
      diffList(a.save.lines, b.save.lines, ls);
      for (const id of ls.keys()) out.add(id);
    }
    if (a.lineStats !== b.lineStats) {
      for (const id in b.lineStats) if (a.lineStats[id] !== b.lineStats[id]) out.add(id);
      for (const id in a.lineStats) if (!(id in b.lineStats)) out.add(id);
    }
    return out;
  });
  const edges = once(() => { const m = new Map<number, readonly [TrackEdge | undefined, TrackEdge | undefined]>(); if (a.edges !== b.edges) diffList(a.edges, b.edges, m); return m; });
  const nodes = once(() => { const m = new Map<number, readonly [TrackNode | undefined, TrackNode | undefined]>(); if (a.nodes !== b.nodes) diffList(a.nodes.values(), b.nodes.values(), m); return m; });
  const d: WorldDiff = {
    get stations() { return stations(); },
    get lines() { return lines(); },
    get edges() { return edges(); },
    get nodes() { return nodes(); },
    links: a.save.links !== b.save.links || a.save.apart !== b.save.apart,
    buildings: a.save.buildings !== b.save.buildings,
  };
  memo.set(b, { from: a, d });
  return d;
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
  return ka.every((k) => sameBuilt((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}
