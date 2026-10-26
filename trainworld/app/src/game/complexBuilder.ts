// Station complexes kept up to date from the stations that changed (T-1380). `ComplexBuilder.update`
// gives exactly what `buildComplexes` (game/complexes.ts) gives for the same stations and links, the
// same numbering included, but works out again only the complexes a change can touch: the old
// complexes of the stations removed, moved or resized, those of the stations new or moved to, and
// those of the stations at the ends of a changed link. Single linkage means no other complex can gain
// or lose a member. `?patch=check` (checks/t1380-complexes.mjs) compares it with a full build.

import { DEMAND_MERGE_M, PLATFORM_GAP_M, buildComplexes, segmentGap, type Complexes } from "./complexes";
import type { StationId, StationInput } from "./types";

type Geo = Pick<StationInput, "id" | "x" | "y" | "heading" | "length">;

interface Comp {
  members: Rec[];
  /** the call that last gave it an index, and the index */
  out: number;
  index: number;
  mark: number;
}

interface Rec {
  id: StationId;
  x: number;
  y: number;
  heading: number;
  length: number;
  /** metres per ground metre at its y */
  k: number;
  /** its platform's two ends */
  ax: number;
  ay: number;
  bx: number;
  by: number;
  /** its position in the last input, and a number it keeps */
  idx: number;
  uid: number;
  /** the call that last saw it; the pass that has it among those worked out again */
  seen: number;
  mark: number;
  parent: Rec;
  comp: Comp | null;
  cell: number;
  gone: boolean;
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
  private out: Complexes | null = null;
  private frameKey = "";
  private lastInput: readonly Geo[] | null = null;
  private linkKeys: string[] = [];
  private linkPairs: (readonly [StationId, StationId])[] = [];
  private calls = 0;
  private stamp = 0;
  private uids = 0;
  /** how the last update went, for the checks: stations worked out again, or -1 for a full build */
  lastWorked = -1;

  /** The complexes of `stations` (in save order) and the player's `links`. `frameKey` names what
   * `kAt` depends on besides y (the world frame's kind): a different one builds everything anew. */
  update(stations: readonly Geo[], links: readonly (readonly [StationId, StationId])[], kAt: (y: number) => number, frameKey: string): Complexes {
    const n = stations.length;
    const call = ++this.calls;
    if (this.out && frameKey !== this.frameKey) this.reset();
    this.frameKey = frameKey;
    const linksChanged = this.diffLinks(links);
    // the very list as last time (the world keeps it while no station changes)
    if (this.out && stations === this.lastInput && !linksChanged.length) {
      this.lastWorked = 0;
      return this.out;
    }
    this.lastInput = stations;
    const before = this.recs.size;

    // what changed
    const added: Rec[] = [], changed: Rec[] = [], changedTo: Geo[] = [], list: Rec[] = new Array(n);
    let survivors = 0, reordered = false, prevIdx = -1;
    for (let i = 0; i < n; i++) {
      const s = stations[i];
      let r = this.recs.get(s.id);
      if (!r) {
        r = this.make(s, kAt);
        this.recs.set(s.id, r);
        added.push(r);
      } else if (r.seen === call) {
        this.reset();
        return this.fallback(stations, links, kAt);
      } else {
        survivors++;
        if (r.idx <= prevIdx) reordered = true;
        prevIdx = r.idx;
        if (r.x !== s.x || r.y !== s.y || r.heading !== s.heading || r.length !== s.length) { changed.push(r); changedTo.push(s); }
      }
      r.seen = call;
      list[i] = r;
    }
    const removed: Rec[] = [];
    if (before > survivors) for (const r of this.recs.values()) if (r.seen !== call) removed.push(r);

    const dirty = added.length + changed.length + removed.length;
    if (reordered || (before > 0 && dirty * REBUILD_SHARE > n + REBUILD_SHARE)) {
      this.reset();
      return this.update(stations, links, kAt, frameKey);
    }
    if (!dirty && !linksChanged.length && this.out) {
      this.lastWorked = 0;
      return this.out;
    }
    for (let i = 0; i < n; i++) list[i].idx = i;
    this.lastWorked = before === 0 ? -1 : dirty;

    // the stations to work out again: `mark` is the pass's number
    const stamp = ++this.stamp;
    const work: Rec[] = [];
    const takeComp = (c: Comp | null) => {
      if (!c || c.mark === stamp) return;
      c.mark = stamp;
      for (const m of c.members) if (!m.gone && m.mark !== stamp) { m.mark = stamp; work.push(m); }
    };
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
    if (links.length && touched.length + removed.length) {
      const by = new Map<StationId, StationId[]>();
      for (const [a, b] of links) { (by.get(a) ?? by.set(a, []).get(a)!).push(b); (by.get(b) ?? by.set(b, []).get(b)!).push(a); }
      for (const r of touched) for (const o of by.get(r.id) ?? []) takeComp(this.recs.get(o)?.comp ?? null);
      for (const r of removed) for (const o of by.get(r.id) ?? []) takeComp(this.recs.get(o)?.comp ?? null);
    }
    for (const id of linksChanged) takeComp(this.recs.get(id)?.comp ?? null);
    // whoever a station new or moved reaches
    for (const r of touched) this.reach(r, takeComp);

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
          if (bucket) for (const o of bucket) if (o.uid > r.uid && o.mark === stamp && (r.idx < o.idx ? linked(r, o) : linked(o, r))) union(r, o);
        }
    }
    for (const [a, b] of links) {
      const i = this.recs.get(a), j = this.recs.get(b);
      if (!i || !j) continue;
      if ((i.mark === stamp) !== (j.mark === stamp)) {
        // a link reaching out of the stations worked out again cannot happen; start over if it does
        this.reset();
        return this.update(stations, links, kAt, frameKey);
      }
      if (i.mark === stamp) union(i, j);
    }
    const byRoot = new Map<Rec, Comp>();
    for (const r of work) {
      const root = find(r);
      let c = byRoot.get(root);
      if (!c) byRoot.set(root, (c = { members: [], out: 0, index: 0, mark: 0 }));
      c.members.push(r);
      r.comp = c;
    }
    return this.renumber(list);
  }

  /** `buildComplexes`'s own numbering: complexes by their first station in the list, stations in it
   * in list order. */
  private renumber(list: Rec[]): Complexes {
    const call = this.calls;
    const of = new Map<StationId, number>(), members: StationId[][] = [];
    for (const r of list) {
      const c = r.comp!;
      if (c.out !== call) { c.out = call; c.index = members.length; members.push([]); }
      members[c.index].push(r.id);
      of.set(r.id, c.index);
    }
    return (this.out = { of, members });
  }

  private reset() {
    this.recs.clear();
    this.grid.clear();
    this.cell = DEMAND_MERGE_M;
    this.out = null;
    this.lastInput = null;
    this.linkKeys = [];
    this.linkPairs = [];
  }

  /** The reference build, for a list this class cannot take (a station twice). */
  private fallback(stations: readonly Geo[], links: readonly (readonly [StationId, StationId])[], kAt: (y: number) => number): Complexes {
    this.lastWorked = -1;
    return buildComplexes(stations, links, kAt);
  }

  private make(s: Geo, kAt: (y: number) => number): Rec {
    const r = { id: s.id, x: 0, y: 0, heading: 0, length: 0, k: 1, ax: 0, ay: 0, bx: 0, by: 0, idx: -1, uid: ++this.uids, seen: 0, mark: 0, comp: null, cell: 0, gone: false } as unknown as Rec;
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
    const h = Math.max(0, s.length) / 2 * r.k, dx = Math.cos(s.heading) * h, dy = Math.sin(s.heading) * h;
    r.ax = s.x - dx;
    r.ay = s.y - dy;
    r.bx = s.x + dx;
    r.by = s.y + dy;
  }

  /** The farthest a station can be linked to another, m. */
  private reachOf(r: Rec): number {
    const h = Math.max(0, r.length) / 2 * r.k;
    return Math.max(DEMAND_MERGE_M * Math.max(1, r.k), 2 * h + PLATFORM_GAP_M * r.k);
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
        if (bucket) for (const o of bucket) if (o !== r && (r.idx < o.idx ? linked(r, o) : linked(o, r))) take(o.comp);
      }
  }

  /** The ids at the ends of the links made or undone since the last call. */
  private diffLinks(links: readonly (readonly [StationId, StationId])[]): StationId[] {
    const keys = links.map(([a, b]) => `${a}\u0000${b}`);
    const out: StationId[] = [];
    if (keys.length !== this.linkKeys.length || keys.some((k, i) => k !== this.linkKeys[i])) {
      const old = new Set(this.linkKeys), now = new Set(keys);
      this.linkPairs.forEach((p, i) => { if (!now.has(this.linkKeys[i])) out.push(p[0], p[1]); });
      links.forEach((p, i) => { if (!old.has(keys[i])) out.push(p[0], p[1]); });
    }
    this.linkKeys = keys;
    this.linkPairs = links.map(([a, b]) => [a, b] as const);
    return out;
  }
}

function key(gx: number, gy: number): number {
  return (gx + CELL_OFFSET) * CELL_SPAN + (gy + CELL_OFFSET);
}

/** Whether `a` and `b` are one complex's: `a` is the earlier in the list, as in `buildComplexes`
 * (the platform gap is not bit-for-bit symmetric). */
function linked(a: Rec, b: Rec): boolean {
  const kk = (a.k + b.k) / 2;
  return Math.hypot(a.x - b.x, a.y - b.y) <= DEMAND_MERGE_M * kk ||
    segmentGap(a.ax, a.ay, a.bx, a.by, b.ax, b.ay, b.bx, b.by) <= PLATFORM_GAP_M * kk;
}
