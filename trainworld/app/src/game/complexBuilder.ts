// Station complexes kept up to date from the stations that changed (T-1380, T-4300). `ComplexBuilder` gives what
// `buildComplexes` (game/complexes.ts) gives for the same stations, links, buildings and separated pairs, numbers
// included, but works out again only the complexes a change can touch: the old complexes of the stations removed,
// moved or resized, those of the stations new or moved to, those at the ends of a link made or undone or of a pair
// separated or let go, and those of a building's members where its members or its outline change. Single linkage
// means no other complex can gain or lose a member. `update` is given every station and finds what changed; `patch`
// only the stations that may have, so neither costs the whole save.
//
// A complex is numbered by its first station's node number, so one no change touched keeps its number. The maps handed
// out are the builder's own, patched in place, with a new handle for each change: a reader compares handles and asks
// `complexChanges` which stations moved. A member list is replaced, never changed. `?patch=check` and
// checks/t1380-complexes.mjs compare it with a whole build.

import { DEMAND_MERGE_M, NO_PAIRS, PLATFORM_GAP_M, buildComplexes, nearEnough, offsetsReach, pairKey, platformOffsets, straightOffsets, type Complexes } from "./complexes";
import { buildingShape, touchesShape, withBuildingLinks, type BuildingShape } from "./buildings";
import type { BuildingInput, StationId, StationInput } from "./types";

type Geo = Pick<StationInput, "id" | "x" | "y" | "heading" | "length" | "line">;
type Pair = readonly [StationId, StationId];

/** What the complexes are made of besides the stations (T-4300), each the same object while it is the same, so a
 * builder looks into one only when it changed: the player's links, the pairs separated (`apartKeys`), the station
 * buildings, the frame's metres per ground metre at a y and the frame's kind. */
export interface ComplexInputs {
  links: readonly Pair[];
  apart: ReadonlySet<string>;
  buildings: readonly BuildingInput[];
  kAt: (y: number) => number;
  frame: string;
}

interface Comp {
  members: Rec[];
  /** its number as handed out: its first member's node number */
  num: number;
  mark: number;
}

interface Rec {
  id: StationId;
  /** its node number: the order stations are taken in */
  num: number;
  x: number;
  y: number;
  heading: number;
  length: number;
  /** metres per ground metre at its y */
  k: number;
  line: ArrayLike<number> | undefined;
  /** its platform as offsets from x, y, drawn (`platformOffsets`) and straight, and how far they reach */
  pts: ArrayLike<number>;
  seg: ArrayLike<number>;
  ext: number;
  /** the call that last saw it; the pass that has it among those worked out again */
  seen: number;
  mark: number;
  parent: Rec;
  comp: Comp | null;
  cell: number;
  gone: boolean;
  /** the buildings it is a member of */
  blds: Bld[] | null;
}

interface Bld {
  corners: readonly number[];
  shape: BuildingShape | null;
  members: Set<Rec>;
}

/** what each handle's update changed: the stations worked out again and those gone, by handle number */
interface Book { next: number; ids: Map<number, ReadonlySet<StationId>> }
/** how many updates back `complexChanges` can say what changed */
const LOG_MAX = 64;
const meta = new WeakMap<Complexes, { book: Book; seq: number }>();
const NONE: ReadonlySet<StationId> = new Set();
const NO_BUILDINGS: readonly BuildingInput[] = [];

/** The stations whose complex (or its number, or its members) may differ from `from` to `to`, when both are handles of
 * one builder at most LOG_MAX updates apart (else null: compare whole). */
export function complexChanges(from: Complexes | null, to: Complexes): ReadonlySet<StationId> | null {
  if (from === to) return NONE;
  const a = from ? meta.get(from) : undefined, b = meta.get(to);
  if (!a || !b || a.book !== b.book || a.seq > b.seq) return null;
  if (b.seq === a.seq + 1) return b.book.ids.get(b.seq) ?? null;
  const out = new Set<StationId>();
  for (let s = a.seq + 1; s <= b.seq; s++) {
    const ids = b.book.ids.get(s);
    if (!ids) return null;
    for (const id of ids) out.add(id);
  }
  return out;
}

/** More stations changed than this share of all: built anew, as `buildComplexes` does. */
const REBUILD_SHARE = 8;
const CELL_OFFSET = 2 ** 21;
const CELL_SPAN = 2 ** 22;

export class ComplexBuilder {
  private recs = new Map<StationId, Rec>();
  private grid = new Map<number, Rec[]>();
  /** side of a grid cell, m: at least as far as two linked stations can be apart */
  private cell = DEMAND_MERGE_M;
  private of = new Map<StationId, number>();
  private members = new Map<number, StationId[]>();
  private out: Complexes | null = null;
  private book: Book = { next: 0, ids: new Map() };
  private frameKey = "";
  private lastInput: readonly Geo[] | null = null;
  private linksIn: readonly Pair[] | null = null;
  private linkKeys: string[] = [];
  private linkPairs: Pair[] = [];
  /** each station's links' other ends */
  private linksBy = new Map<StationId, StationId[]>();
  /** T-4048: the pairs the distance rule does not join, as last given */
  private apart: ReadonlySet<string> = NO_PAIRS;
  private bldIn: readonly BuildingInput[] | null = null;
  private blds = new Map<number, Bld>();
  private calls = 0;
  private stamp = 0;
  /** how the last update went, for the checks: stations worked out again, or -1 for a full build */
  lastWorked = -1;

  /** T-3750: building `id`'s first member (by node number), as the last call left them; undefined for none. */
  buildingFirst(id: number): StationId | undefined {
    let first: Rec | undefined;
    for (const r of this.blds.get(id)?.members ?? []) if (!first || r.num < first.num) first = r;
    return first?.id;
  }

  /** Whether `patch` can take changes to the last call's stations in frame `frameKey`. */
  ready(frameKey: string): boolean {
    return !!this.out && frameKey === this.frameKey;
  }

  /** The complexes of `stations` (each once, any order, ids node numbers), the player's `links`, the pairs kept `apart`
   * (`apartKeys`, T-4048) and the station `buildings` (T-2217). `frameKey` names what `kAt` depends on besides y (the
   * world frame's kind): a different one builds everything anew. */
  update(stations: readonly Geo[], links: readonly Pair[], kAt: (y: number) => number, frameKey: string, apart: ReadonlySet<string> = NO_PAIRS, buildings: readonly BuildingInput[] = NO_BUILDINGS): Complexes {
    const call = ++this.calls;
    if (this.out && frameKey !== this.frameKey) this.reset();
    this.frameKey = frameKey;
    // the very list as last time (the world keeps it while no station changes)
    if (this.out && stations === this.lastInput && this.sameInputs(links, apart, buildings)) {
      this.lastWorked = 0;
      return this.out;
    }
    const n = stations.length, before = this.recs.size;
    const added: Rec[] = [], changed: Rec[] = [], changedTo: Geo[] = [];
    let survivors = 0;
    for (const s of stations) {
      let r = this.recs.get(s.id);
      if (!r) {
        r = this.make(s, kAt);
        this.recs.set(s.id, r);
        added.push(r);
      } else if (r.seen === call) {
        this.reset();
        return this.fallback(stations, links, kAt, apart, buildings);
      } else {
        survivors++;
        if (moved(r, s)) { changed.push(r); changedTo.push(s); }
      }
      r.seen = call;
    }
    const removed: Rec[] = [];
    if (before > survivors) for (const r of this.recs.values()) if (r.seen !== call) removed.push(r);
    const dirty = added.length + changed.length + removed.length;
    if (before > 0 && dirty * REBUILD_SHARE > n + REBUILD_SHARE) {
      this.reset();
      return this.update(stations, links, kAt, frameKey, apart, buildings);
    }
    this.lastInput = stations;
    if (!dirty && this.out && this.sameInputs(links, apart, buildings)) {
      this.lastWorked = 0;
      return this.out;
    }
    this.lastWorked = before === 0 ? -1 : dirty;
    return this.apply(added, changed, changedTo, removed, links, kAt, apart, buildings);
  }

  /** The complexes after the stations `changes` names (each as it is now, null where gone) changed from the last call's,
   * every other station as it was; the rest as `update`. Only after an `update` in this frame (`ready`). */
  patch(changes: Iterable<readonly [StationId, Geo | null]>, links: readonly Pair[], kAt: (y: number) => number, frameKey: string, apart: ReadonlySet<string> = NO_PAIRS, buildings: readonly BuildingInput[] = NO_BUILDINGS): Complexes {
    if (!this.ready(frameKey)) throw new Error("ComplexBuilder.patch before a whole update in this frame");
    const call = ++this.calls;
    this.lastInput = null;
    const added: Rec[] = [], changed: Rec[] = [], changedTo: Geo[] = [], removed: Rec[] = [];
    for (const [id, s] of changes) {
      const r = this.recs.get(id);
      if (r?.seen === call) continue;
      if (!s) {
        if (r) { r.seen = call; removed.push(r); }
      } else if (!r) {
        const m = this.make(s, kAt);
        m.seen = call;
        this.recs.set(s.id, m);
        added.push(m);
      } else {
        r.seen = call;
        if (moved(r, s)) { changed.push(r); changedTo.push(s); }
      }
    }
    const dirty = added.length + changed.length + removed.length;
    if (!dirty && this.sameInputs(links, apart, buildings)) {
      this.lastWorked = 0;
      return this.out!;
    }
    this.lastWorked = dirty;
    return this.apply(added, changed, changedTo, removed, links, kAt, apart, buildings);
  }

  private sameInputs(links: readonly Pair[], apart: ReadonlySet<string>, buildings: readonly BuildingInput[]): boolean {
    return links === this.linksIn && apart === this.apart && buildings === this.bldIn;
  }

  private apply(added: Rec[], changed: Rec[], changedTo: Geo[], removed: Rec[], links: readonly Pair[], kAt: (y: number) => number, apart: ReadonlySet<string>, buildings: readonly BuildingInput[]): Complexes {
    // the stations to work out again: `mark` is the pass's number
    const stamp = ++this.stamp;
    const work: Rec[] = [], taken: Comp[] = [];
    const takeComp = (c: Comp | null) => {
      if (!c || c.mark === stamp) return;
      c.mark = stamp;
      taken.push(c);
      for (const m of c.members) if (!m.gone && m.mark !== stamp) { m.mark = stamp; work.push(m); }
    };
    const ends = this.diffLinks(links);
    // T-4048: the ends of a pair kept apart or let join again are worked out again, as a link's
    if (apart !== this.apart) {
      for (const k of apart) if (!this.apart.has(k)) ends.push(...(k.split("|") as StationId[]));
      for (const k of this.apart) if (!apart.has(k)) ends.push(...(k.split("|") as StationId[]));
      this.apart = apart;
    }
    for (const r of removed) r.gone = true;
    for (const r of removed) {
      this.gridRemove(r);
      this.recs.delete(r.id);
      takeComp(r.comp);
    }
    for (const r of changed) {
      takeComp(r.comp);
      this.gridRemove(r);
    }
    const touched = changed.concat(added);
    changed.forEach((r, i) => this.fill(r, changedTo[i], kAt));
    this.place(touched);
    for (const r of added) if (r.mark !== stamp) { r.mark = stamp; work.push(r); }
    for (const r of changed) if (r.mark !== stamp) { r.mark = stamp; work.push(r); }
    // the other ends of links a changed station is on, the ends of links made or undone
    if (this.linksBy.size) {
      for (const r of touched) for (const o of this.linksBy.get(r.id) ?? []) takeComp(this.recs.get(o)?.comp ?? null);
      for (const r of removed) for (const o of this.linksBy.get(r.id) ?? []) takeComp(this.recs.get(o)?.comp ?? null);
    }
    for (const id of ends) takeComp(this.recs.get(id)?.comp ?? null);
    this.syncBuildings(buildings, kAt, touched, removed, takeComp);
    // whoever a station new or moved reaches
    for (const r of touched) this.reach(r, takeComp);
    // (links or buildings given anew as they were: the same complexes, the same handle)
    if (!work.length && !removed.length && this.out) return this.out;

    // single linkage among them
    for (const r of work) r.parent = r;
    const find = (r: Rec): Rec => {
      while (r.parent !== r) { r.parent = r.parent.parent; r = r.parent; }
      return r;
    };
    const union = (a: Rec, b: Rec) => {
      const ra = find(a), rb = find(b);
      if (ra !== rb) ra.parent = rb;
    };
    const cell = this.cell;
    for (const r of work) {
      const cx = Math.floor(r.x / cell), cy = Math.floor(r.y / cell);
      for (let gx = cx - 1; gx <= cx + 1; gx++)
        for (let gy = cy - 1; gy <= cy + 1; gy++) {
          const bucket = this.grid.get(key(gx, gy));
          if (bucket) for (const o of bucket) if (o.num > r.num && o.mark === stamp && linked(r, o) && !this.isApart(r, o)) union(r, o);
        }
    }
    // a link or a building reaching out of the stations worked out again cannot happen; start over if one does
    for (const [a, b] of this.linkPairs) {
      const i = this.recs.get(a), j = this.recs.get(b);
      if (!i || !j) continue;
      if ((i.mark === stamp) !== (j.mark === stamp)) return this.rebuild(kAt);
      if (i.mark === stamp) union(i, j);
    }
    for (const b of this.blds.values()) {
      let first: Rec | null = null, out = false;
      for (const m of b.members) {
        if (m.mark !== stamp) out = true;
        else if (first) union(m, first);
        else first = m;
      }
      if (first && out) return this.rebuild(kAt);
    }
    const byRoot = new Map<Rec, Comp>(), made: Comp[] = [];
    for (const r of work) {
      const root = find(r);
      let c = byRoot.get(root);
      if (!c) { byRoot.set(root, (c = { members: [], num: 0, mark: 0 })); made.push(c); }
      c.members.push(r);
      r.comp = c;
    }
    // the maps patched: the complexes taken apart go, those made come, each worked out station takes its complex's number
    for (const c of taken) this.members.delete(c.num);
    for (const c of made) {
      if (c.members.length > 1) c.members.sort((p, q) => p.num - q.num);
      c.num = c.members[0].num;
      this.members.set(c.num, c.members.map((r) => r.id));
    }
    const ids = new Set<StationId>();
    for (const r of work) { this.of.set(r.id, r.comp!.num); ids.add(r.id); }
    for (const r of removed) { this.of.delete(r.id); ids.add(r.id); }
    return this.hand(ids);
  }

  /** A new handle on the maps, with what this update changed for `complexChanges`. */
  private hand(ids: ReadonlySet<StationId>): Complexes {
    const out: Complexes = { of: this.of, members: this.members }, book = this.book, seq = book.next++;
    if (seq > 0) {
      book.ids.set(seq, ids);
      book.ids.delete(seq - LOG_MAX);
    }
    meta.set(out, { book, seq });
    return (this.out = out);
  }

  /** Every station built anew from what the builder holds (as given last, the changes in). */
  private rebuild(kAt: (y: number) => number): Complexes {
    const geo: Geo[] = [];
    for (const r of this.recs.values()) if (!r.gone) geo.push({ id: r.id, x: r.x, y: r.y, heading: r.heading, length: r.length, ...(r.line ? { line: r.line as Float32Array } : {}) });
    const links = this.linksIn ?? [], apart = this.apart, buildings = this.bldIn ?? NO_BUILDINGS, frameKey = this.frameKey;
    this.reset();
    return this.update(geo, links, kAt, frameKey, apart, buildings);
  }

  private reset() {
    this.recs.clear();
    this.grid.clear();
    this.cell = DEMAND_MERGE_M;
    // (new maps: the handles out keep the old ones as they were)
    this.of = new Map();
    this.members = new Map();
    this.out = null;
    this.book = { next: 0, ids: new Map() };
    this.lastInput = null;
    this.linksIn = null;
    this.linkKeys = [];
    this.linkPairs = [];
    this.linksBy = new Map();
    this.apart = NO_PAIRS;
    this.bldIn = null;
    this.blds.clear();
  }

  /** The reference build, for a list this class cannot take (a station twice). */
  private fallback(stations: readonly Geo[], links: readonly Pair[], kAt: (y: number) => number, apart: ReadonlySet<string>, buildings: readonly BuildingInput[]): Complexes {
    this.lastWorked = -1;
    return buildComplexes(stations, withBuildingLinks(links, buildings, stations, kAt), kAt, apart);
  }

  private isApart(a: Rec, b: Rec): boolean {
    return this.apart.size > 0 && this.apart.has(pairKey(a.id, b.id));
  }

  private make(s: Geo, kAt: (y: number) => number): Rec {
    const r = { id: s.id, num: Number(s.id), x: 0, y: 0, heading: 0, length: 0, k: 1, line: undefined, pts: [], seg: [], ext: 0, seen: 0, mark: 0, comp: null, cell: 0, gone: false, blds: null } as unknown as Rec;
    r.parent = r;
    this.fill(r, s, kAt);
    return r;
  }

  private fill(r: Rec, s: Geo, kAt: (y: number) => number) {
    r.x = s.x;
    r.y = s.y;
    r.heading = s.heading;
    r.length = s.length;
    r.k = kAt(s.y);
    r.line = s.line;
    r.seg = straightOffsets(s, r.k);
    r.pts = platformOffsets(s, r.k, r.seg);
    r.ext = Math.max(offsetsReach(r.pts), offsetsReach(r.seg));
  }

  /** The farthest a station can be linked to another, m. */
  private reachOf(r: Rec): number {
    return Math.max(DEMAND_MERGE_M * Math.max(1, r.k), 2 * r.ext + PLATFORM_GAP_M * r.k);
  }

  /** Put the recs in the grid, which is made larger first if one of them needs it. */
  private place(recs: Rec[]) {
    let need = this.cell;
    for (const r of recs) need = Math.max(need, this.reachOf(r));
    if (need > this.cell) {
      this.cell = need;
      this.grid.clear();
      for (const r of this.recs.values()) this.gridAdd(r);
    } else for (const r of recs) this.gridAdd(r);
  }

  private gridAdd(r: Rec) {
    r.cell = key(Math.floor(r.x / this.cell), Math.floor(r.y / this.cell));
    const b = this.grid.get(r.cell);
    if (b) b.push(r);
    else this.grid.set(r.cell, [r]);
  }

  private gridRemove(r: Rec) {
    const b = this.grid.get(r.cell);
    if (!b) return;
    const at = b.indexOf(r);
    if (at >= 0) b.splice(at, 1);
    if (!b.length) this.grid.delete(r.cell);
  }

  /** `take` for each complex of a station `r` is linked to. */
  private reach(r: Rec, take: (c: Comp | null) => void) {
    const cx = Math.floor(r.x / this.cell), cy = Math.floor(r.y / this.cell);
    for (let gx = cx - 1; gx <= cx + 1; gx++)
      for (let gy = cy - 1; gy <= cy + 1; gy++) {
        const bucket = this.grid.get(key(gx, gy));
        if (bucket) for (const o of bucket) if (o !== r && (r.num < o.num ? linked(r, o) : linked(o, r)) && !this.isApart(r, o)) take(o.comp);
      }
  }

  /** The ids at the ends of the links made or undone since the last call. */
  private diffLinks(links: readonly Pair[]): StationId[] {
    if (links === this.linksIn) return [];
    this.linksIn = links;
    const keys = links.map(([a, b]) => `${a}|${b}`);
    const out: StationId[] = [];
    if (keys.length !== this.linkKeys.length || keys.some((k, i) => k !== this.linkKeys[i])) {
      const old = new Set(this.linkKeys), now = new Set(keys);
      this.linkPairs.forEach((p, i) => { if (!now.has(this.linkKeys[i])) out.push(p[0], p[1]); });
      links.forEach((p, i) => { if (!old.has(keys[i])) out.push(p[0], p[1]); });
      this.linksBy = new Map();
      for (const [a, b] of links) { (this.linksBy.get(a) ?? this.linksBy.set(a, []).get(a)!).push(b); (this.linksBy.get(b) ?? this.linksBy.set(b, []).get(b)!).push(a); }
    }
    this.linkKeys = keys;
    this.linkPairs = links.map(([a, b]) => [a, b] as const);
    return out;
  }

  /** T-2217: each building's members kept: an outline new or moved finds its members among every station, a station
   * new, moved or gone looks at the outlines alone. The complexes of members that join or leave a building are worked out
   * again (those a building loses were in the leaving station's own). */
  private syncBuildings(buildings: readonly BuildingInput[], kAt: (y: number) => number, touched: Rec[], removed: Rec[], take: (c: Comp | null) => void) {
    const fresh: Bld[] = [];
    const leave = (b: Bld, r: Rec) => {
      b.members.delete(r);
      if (r.blds) { const v = r.blds.filter((x) => x !== b); r.blds = v.length ? v : null; }
    };
    if (buildings !== this.bldIn) {
      const now = new Map(buildings.map((b) => [b.id, b] as const));
      for (const [id, b] of this.blds) {
        const c = now.get(id)?.corners;
        if (c && (c === b.corners || sameNums(c, b.corners))) { b.corners = c; continue; }
        for (const m of [...b.members]) { take(m.comp); leave(b, m); }
        this.blds.delete(id);
      }
      for (const b of buildings) if (!this.blds.has(b.id)) { const x: Bld = { corners: b.corners, shape: buildingShape(b, kAt), members: new Set() }; this.blds.set(b.id, x); fresh.push(x); }
      this.bldIn = buildings;
    }
    for (const r of removed) if (r.blds) for (const b of r.blds) b.members.delete(r);
    for (const b of fresh) {
      if (!b.shape) continue;
      for (const r of this.recs.values()) if (touchesShape(b.shape, r, r.pts)) { b.members.add(r); (r.blds ??= []).push(b); }
      for (const m of b.members) take(m.comp);
    }
    if (!this.blds.size) return;
    for (const r of touched) {
      for (const b of this.blds.values()) {
        if (!b.shape || fresh.includes(b)) continue;
        const was = b.members.has(r), now = touchesShape(b.shape, r, r.pts);
        if (was === now) continue;
        if (!now) { leave(b, r); continue; }
        b.members.add(r);
        (r.blds ??= []).push(b);
        for (const m of b.members) take(m.comp);
      }
    }
  }
}

function key(gx: number, gy: number): number {
  return (gx + CELL_OFFSET) * CELL_SPAN + (gy + CELL_OFFSET);
}

/** Whether station `s` is placed or shaped otherwise than `r` holds it. */
function moved(r: Rec, s: Geo): boolean {
  return r.x !== s.x || r.y !== s.y || r.heading !== s.heading || r.length !== s.length || (r.line !== s.line && !sameLine(r.line, s.line));
}

const sameLine = (p: ArrayLike<number> | undefined, q: ArrayLike<number> | undefined) => !!p && !!q && sameNums(p, q);
function sameNums(p: ArrayLike<number>, q: ArrayLike<number>): boolean {
  if (p.length !== q.length) return false;
  for (let i = 0; i < p.length; i++) if (!Object.is(p[i], q[i])) return false;
  return true;
}

/** Whether `a` and `b` are one complex's: `a` is the earlier in node order, as in `buildComplexes`
 * (the platform gap is not bit-for-bit symmetric). */
function linked(a: Rec, b: Rec): boolean {
  return nearEnough(a, a.pts, a.seg, b, b.pts, b.seg, (a.k + b.k) / 2);
}

/** `?patch=check`: `got` against a whole build of the same stations and inputs; what differs, or null. */
export function complexesDiffer(got: Complexes, stations: readonly Geo[], links: readonly Pair[], kAt: (y: number) => number, apart: ReadonlySet<string> = NO_PAIRS, buildings: readonly BuildingInput[] = NO_BUILDINGS): string | null {
  const want = buildComplexes(stations, withBuildingLinks(links, buildings, stations, kAt), kAt, apart);
  if (got.of.size !== want.of.size || got.members.size !== want.members.size) return `sizes ${got.of.size}/${got.members.size} against ${want.of.size}/${want.members.size}`;
  for (const [id, c] of want.of) if (got.of.get(id) !== c) return `station ${id} in ${got.of.get(id)} against ${c}`;
  for (const [c, m] of want.members) if (got.members.get(c)?.join() !== m.join()) return `complex ${c}: ${got.members.get(c)?.join()} against ${m.join()}`;
  return null;
}
